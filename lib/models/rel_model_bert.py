"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
#from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size, nms_overlaps
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, load_resnet
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.pytorch_misc import random_choose
import math
from lib.attention.bert import BERT
from lib.utils.prepare_feat_bert import prepare_feat, postdiso_feat
def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds[-1] + 1
    rois_per_image = scores.new(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


MODES = ('sgdet', 'sgcls', 'predcls', 'predcls_nongtbox', 'detclass')


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """
    def __init__(self, classes, rel_classes, mode='sgdet',
                 embed_dim=200, obj_dim=2048,
                 dim_obj_hidden=256,
                 nl_obj=12,
                 nh_obj=12,
                 dim_edge_hidden=256,
                 nl_edge=12,
                 nh_edge=12,
                 dropout_rate=0.2):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode

        self.embed_dim = embed_dim
        self.obj_dim = obj_dim

        self.dim_obj_hidden = dim_obj_hidden
        self.nl_obj = nl_obj
        self.nh_obj = nh_obj

        self.dim_edge_hidden = dim_edge_hidden
        self.nl_edge = nl_edge
        self.nh_edge = nh_edge

        self.dropout_rate = dropout_rate
        # EMBEDDINGS
        #self.leakyrelu = nn.LeakyReLU(self.alpha)
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)

        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed_in_edge = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed_in_edge.weight.data = embed_vecs.clone()
        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])


        if self.nl_obj > 0 and self.mode != 'predcls_nongtbox':
            obj_ctx_indim = self.obj_dim + self.embed_dim + 128
            self.obj_ctx_bert = BERT(
                input_dim=obj_ctx_indim,
                hidden_dim=self.dim_obj_hidden,
                n_layers=self.nl_obj,
                attn_heads=self.nh_obj,
                dropout=self.dropout_rate)

        self.obj_ctx_classifier =  nn.Linear(self.dim_obj_hidden, self.num_classes)

        if self.mode == 'detclass':
            return
        if self.nl_edge > 0:
            #print('input_dim: ',input_dim)
            edge_ctx_indim = self.obj_dim + self.embed_dim + 128
            self.edge_ctx_bert = BERT(
                input_dim=edge_ctx_indim,
                hidden_dim=self.dim_edge_hidden,
                n_layers=self.nl_edge,
                attn_heads=self.nh_edge,
                dropout=self.dropout_rate)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)


    def edge_ctx(self, obj_feats, im_inds, obj_preds):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_feats_t, mask = prepare_feat(obj_feats, im_inds.data)
        edge_reps = self.edge_ctx_bert(obj_feats_t, mask)
        edge_reps = postdiso_feat(edge_reps, im_inds.data)
        # now we're good! unperm
        return edge_reps

    def obj_ctx(self, obj_feats, obj_dists, obj_labels=None,
                boxes_per_cls=None,im_inds=None,
                ):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        prior_obj_dists = obj_dists
        obj_feats_t, mask = prepare_feat(obj_feats, im_inds.data)
        obj_ctx_rep = self.obj_ctx_bert(obj_feats_t, mask)
        obj_ctx_rep = postdiso_feat(obj_ctx_rep, im_inds.data)
        obj_dists = self.obj_ctx_classifier(obj_ctx_rep)

        if prior_obj_dists is not None:
            obj_dists = obj_dists + prior_obj_dists
        boxes_for_nms = boxes_per_cls

        if self.training:
            nms_labels = obj_labels
            nonzero_pred = obj_dists[:, 1:].max(1)[1] + 1
            is_bg = (nms_labels.data == 0).nonzero()
            if is_bg.dim() > 0:
                nms_labels[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
        else:
            out_dist_sample = F.softmax(obj_dists, dim=1)

            nms_labels = out_dist_sample[:, 1:].max(1)[1] + 1

        if (boxes_for_nms is not None and not self.training):
            is_overlap = nms_overlaps(boxes_for_nms.data).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists = obj_dists
            out_dists_sampled = F.softmax(out_dists, 1).data.cpu().numpy()
            out_dists_sampled[:, 0] = 0

            nms_labels = nms_labels[0].data.new(len(nms_labels)).fill_(0)

            for i in range(nms_labels.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                nms_labels[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            output_labels = Variable(nms_labels)
        else:
            output_labels = nms_labels
        return obj_ctx_rep, obj_dists, output_labels


    def get_union_box(self, rois, union_inds):
        im_inds = rois[:, 0][union_inds[:, 0]]

        union_rois = torch.cat((
            im_inds[:, None],
            torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
            torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
        ), 1)
        return union_rois

    def max_pooling_image(self, obj_dist, num_box):
        output = []
        pre_i = 0
        for i in num_box.data.cpu().numpy():
            i = int(i)
            output.append(((obj_dist[pre_i:pre_i+i].max(0)[0])).clone())
            pre_i = i
        return torch.stack(output)

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None,
                box_priors=None, boxes_per_cls=None):
        """
        Forward pass through the object and edge context
        :param obj_fmaps: shape: [num_boxes, dim_feature]
        :param obj_logits: shape: [num_boxes, num_classes]  before softmax
        :param im_inds: shape: [num_boxes, 1]  each is img_ind
        :param obj_labels: shape: [num_boxes, 1]  each is box class
        :param box_priors: shape: [num_boxes, 4]  each is box position
        :return:
        """
        obj_logits_softmax = F.softmax(obj_logits, dim=1)
        #if self.mode == 'predcls':
        #   obj_logits = Variable(to_onehot(obj_labels.data, self.num_classes))
        #   obj_logits_softmax = obj_logits


        obj_embed = obj_logits_softmax @ self.obj_embed.weight
        obj_embed = F.dropout(obj_embed, self.dropout_rate, training=self.training)
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))

        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)


        if self.nl_obj > 0:

            obj_ctx, obj_dists2, obj_preds = self.obj_ctx(
                obj_feats = obj_pre_rep,
                obj_dists = obj_logits,
                obj_labels = obj_labels,
                boxes_per_cls = boxes_per_cls,
                im_inds=im_inds,
            )

        else:
            # UNSURE WHAT TO DO HERE
            if self.mode == 'predcls':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))
            else:
                obj_dists2 = self.decoder_lin(obj_pre_rep)

            if self.mode == 'sgdet' and not self.training:
                # NMS here for baseline

                probs = F.softmax(obj_dists2, 1)
                nms_mask = obj_dists2.data.clone()
                nms_mask.zero_()
                for c_i in range(1, obj_dists2.size(1)):
                    scores_ci = probs.data[:, c_i]
                    boxes_ci = boxes_per_cls.data[:, c_i]

                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

                obj_preds = Variable(nms_mask * probs.data, volatile=True)[:,1:].max(1)[1] + 1
            else:
                obj_preds = obj_labels if obj_labels is not None else obj_dists2[:,1:].max(1)[1] + 1
            obj_ctx = obj_pre_rep


        if self.nl_edge > 0:
            obj_embed_to_edge =  self.obj_embed_in_edge(obj_preds)
            edge_pre_rep = torch.cat((obj_fmaps, obj_embed_to_edge, pos_embed), 1)
            edge_ctx = self.edge_ctx(
                obj_feats = edge_pre_rep,
                obj_preds = obj_preds, #obj_preds_zeros, #obj_preds_zeros obj_preds_nozeros
                im_inds=im_inds,
            )

        else:
            edge_ctx = obj_ctx

        return obj_dists2, obj_preds, edge_ctx


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """
    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=12, nl_edge=12, nh_obj=12, nh_edge=12,
                 use_resnet=False, thresh=0.01,
                 use_proposals=False,
                 rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True,
                 bg_num_graph=-1, bg_num_rel=-1,
                 fb_thr=0.5, with_biliner_score=False,
                 nms_union=False,
                 ext_feat=False, with_gt_adj_mat=False,
                 *args, **kwargs,
                 ):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: Whether to use vision in the final product
        :param require_overlap_det: Whether two objects must intersect
        :param embed_dim: Dimension for all embeddings
        :param hidden_dim: LSTM hidden size
        :param obj_dim:
        """
        super(RelModel, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode

        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.pooling_dim = pooling_dim
        self.use_resnet = use_resnet
        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision=limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.bg_num_graph = bg_num_graph
        self.bg_num_rel = bg_num_rel
        self.fb_thr = fb_thr
        self.dropout_rate = rec_dropout
        self.with_biliner_score = with_biliner_score
        self.nl_obj = nl_obj
        self.nh_obj = nh_obj
        self.nl_edge = nl_edge
        self.nh_edge = nh_edge
        self.nms_union = nms_union
        self.with_gt_adj_mat = with_gt_adj_mat
        self.with_adaptive = False

        self.ext_feat = ext_feat

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' or mode == 'predcls_nongtbox' else 'gtbox',
            thresh=thresh,
            max_per_img=64,
            bg_num_graph=self.bg_num_graph,
            bg_num_rel=self.bg_num_rel,
            with_gt_adj_mat=with_gt_adj_mat,
            backbone_type=('resnet' if use_resnet else 'vgg')
        )
        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim,
                                         obj_dim=self.obj_dim,
                                         dim_obj_hidden=self.hidden_dim,
                                         nl_obj=self.nl_obj,
                                         nh_obj=self.nh_obj,
                                         dim_edge_hidden=self.hidden_dim,
                                         nl_edge=self.nl_edge,
                                         nh_edge=self.nh_edge,
                                         dropout_rate=self.dropout_rate
                                         )

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        #self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
        #                                      dim=1024 if use_resnet else 512)

        if use_resnet:

            roi_fmap = load_resnet(pretrained=False)[1]
            if pooling_dim != 2048:
                roi_fmap.append(nn.Linear(2048, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_resnet(pretrained=False)[1]
            self.compress_union = None
            self.compress = None

        else:

            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
            self.compress_union = None
            self.compress = None



        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512, compress_union=self.compress_union)

        ###################################
        post_lstm_in_dim = self.hidden_dim

        if self.mode == 'detclass':
            return
        self.post_obj_lstm = nn.Linear(post_lstm_in_dim, self.pooling_dim)
        self.post_sub_lstm = nn.Linear(post_lstm_in_dim, self.pooling_dim)
        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
        # (Half contribution comes from LSTM, half from embedding.

        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_obj_lstm.weight = torch.nn.init.xavier_normal(self.post_obj_lstm.weight, gain=1.0)
        self.post_sub_lstm.weight = torch.nn.init.xavier_normal(self.post_sub_lstm.weight, gain=1.0)
        '''
        self.post_obj_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_obj_lstm.bias.data.zero_()
        self.post_sub_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
        self.post_sub_lstm.bias.data.zero_()
        '''

        if nl_edge == 0:
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim*2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))


        if with_biliner_score:
            self.rel_bilinear = nn.Linear(self.pooling_dim, self.num_rels, bias=False)
            self.rel_bilinear.weight = torch.nn.init.xavier_normal(self.rel_bilinear.weight, gain=1.0)
        else:
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
            self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias:
            self.freq_bias = FrequencyBias(with_bg=bg_num_rel!=0)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def visual_rep(self, features, rois, pair_inds):
        """
        Classify the features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds inds to use when predicting
        :return: score_pred, a [num_rois, num_classes] array
                 box_pred, a [num_rois, num_classes, 4] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        if not self.use_resnet:
            return self.roi_fmap(uboxes)
        else:
            #print('uboxes: ',uboxes.size())
            roi_fmap_t = self.roi_fmap(uboxes)
            #print('roi_fmap_t: ',roi_fmap_t.size())
            return roi_fmap_t.mean(3).mean(2)

    def get_rel_inds(self, rel_labels, rel_labels_offset, im_inds, box_priors):
        # Get the relationship candidates
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
            rel_inds_offset = rel_labels_offset[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)
                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
            rel_inds_offset =  rel_inds
        return rel_inds, rel_inds_offset

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        '''
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
                        self.compress(features) if self.use_resnet else features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))
        '''

        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
                       features, rois)
        if not self.use_resnet:
            return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))
        else:
            return self.roi_fmap_obj(feature_pool).mean(3).mean(2)



    def bilinear_score(self, sub, obj):
        '''

        Args:
            obj:  num_rel, dim_hid
            sub: num_rel, dim_hid

        Returns:

        '''
        prod_rep_graph = sub * obj
        rel_dists_graph = self.rel_bilinear(prod_rep_graph)
        return rel_dists_graph

    def adp_bilinear_score(self, sub, obj):
        '''

        Args:
            obj:  num_rel, dim_hid
            sub: num_rel, dim_hid

        Returns:

        '''
        prod_rep_graph = sub * obj
        rel_dists_graph = self.adp_bilinear(prod_rep_graph)
        return rel_dists_graph

    def forward(self, ids, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_adj_mat=None, gt_rels=None,
                gt_mul_label=None, gt_mul_rel=None, gt_mul_label_num=None,
                num_box=None,
                proposals=None,
                ae_adj_mat_rel=None,
                ae_adj_mat_obj=None,
                ae_pred_rel_inds=None,
                train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """

        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if self.mode == 'detclass' and self.nl_obj == 0 and not self.with_gcn:
            if self.training:
                return result
            else:
                rm_obj_softmax = F.softmax(result.rm_obj_dists, dim=-1)
                obj_preds = rm_obj_softmax[:, 1:].max(-1)[1] + 1
                twod_inds = arange(obj_preds.data) * self.num_classes + obj_preds.data
                obj_scores = F.softmax(result.rm_obj_dists, dim=-1)
                obj_preds = obj_preds.cpu().data.numpy()
                rm_box_priors = result.rm_box_priors.cpu().data.numpy()
                return rm_box_priors, obj_preds, obj_scores

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        obj_feat_im_inds = im_inds
        boxes = result.rm_box_priors

        if self.mode == 'sgdet':

            max_num_bg = int(max(self.bg_num_graph, self.bg_num_rel))
            if self.bg_num_graph == -1 or self.bg_num_rel == -1:
                max_num_bg = -1
            rel_labels_fg, result.gt_adj_mat_graph, rel_labels_offset_fg \
                = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                  gt_boxes.data, gt_classes.data, gt_rels.data,
                                  image_offset, filter_non_overlap=True,
                                  num_sample_per_gt=1, max_obj_num=self.max_obj_num,
                                  time_bg=0, time_fg=1)
            rel_labels_bg, result.gt_adj_mat_rel, rel_labels_offset_bg \
                = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                  gt_boxes.data, gt_classes.data, gt_rels.data,
                                  image_offset, filter_non_overlap=True,
                                  num_sample_per_gt=1, max_obj_num=self.max_obj_num,
                                  time_bg=max_num_bg, time_fg=0)
            max_num_bg = (rel_labels_bg.shape[0] / (1.0 * rel_labels_fg.shape[0]))
            num_bg_graph = int((self.bg_num_graph / (1.0 * max_num_bg)) * rel_labels_bg.shape[0])
            num_bg_rel = int((self.bg_num_rel / (1.0 * max_num_bg)) * rel_labels_bg.shape[0])
            if self.bg_num_graph == -1:
                num_bg_graph = rel_labels_bg.shape[0]
            if self.bg_num_rel == -1:
                num_bg_rel = rel_labels_bg.shape[0]

            if num_bg_graph > 0:

                if num_bg_graph < rel_labels_bg.size(0):
                    rel_labels_bg_ind_graph = random_choose(rel_labels_bg, num_bg_graph, re_id=True)
                    rel_labels_graph_bg_ch = rel_labels_bg[rel_labels_bg_ind_graph].contiguous()
                    rel_labels_graph_offset_bg_ch = rel_labels_offset_bg[rel_labels_bg_ind_graph].contiguous()
                else:
                    rel_labels_graph_bg_ch = rel_labels_bg
                    rel_labels_graph_offset_bg_ch = rel_labels_offset_bg
                rel_labels_graph = torch.cat([rel_labels_fg, rel_labels_graph_bg_ch], 0)
                rel_labels_offset_graph = torch.cat([rel_labels_offset_fg, rel_labels_graph_offset_bg_ch], 0)
            else:
                rel_labels_graph = rel_labels_fg
                rel_labels_offset_graph = rel_labels_offset_fg
            if num_bg_rel > 0:
                if num_bg_rel < rel_labels_bg.size(0):
                    rel_labels_bg_ind_rel = random_choose(rel_labels_bg, num_bg_rel, re_id=True)
                    rel_labels_rel_bg_ch = rel_labels_bg[rel_labels_bg_ind_rel].contiguous()
                    rel_labels_rel_offset_bg_ch = rel_labels_offset_bg[rel_labels_bg_ind_rel].contiguous()
                else:
                    rel_labels_rel_bg_ch = rel_labels_bg
                    rel_labels_rel_offset_bg_ch = rel_labels_offset_bg
                rel_labels_rel = torch.cat([rel_labels_fg, rel_labels_rel_bg_ch], 0)
                rel_labels_offset_rel = torch.cat([rel_labels_offset_fg, rel_labels_rel_offset_bg_ch], 0)
            else:
                rel_labels_rel = rel_labels_fg
                rel_labels_offset_rel = rel_labels_offset_fg

            result.rel_labels_rel = rel_labels_rel
            result.rel_labels_offset_rel = rel_labels_offset_rel
            result.rel_labels_graph = rel_labels_graph
            result.rel_labels_offset_graph = rel_labels_offset_graph
            _, perm_rel = torch.sort(result.rel_labels_rel[:, 0] * (boxes.size(0) ** 2)
                                     + result.rel_labels_rel[:, 1] * boxes.size(0)
                                     + result.rel_labels_rel[:, 2])

            result.rel_labels_rel = result.rel_labels_rel[perm_rel]
            result.rel_labels_offset_rel = result.rel_labels_offset_rel[perm_rel]

            _, perm_graph = torch.sort(result.rel_labels_graph[:, 0] * (boxes.size(0) ** 2)
                                       + result.rel_labels_graph[:, 1] * boxes.size(0)
                                       + result.rel_labels_graph[:, 2])
            result.rel_labels_graph = result.rel_labels_graph[perm_graph]
            result.rel_labels_offset_graph = result.rel_labels_offset_graph[perm_graph]

            num_true_rel = rel_labels_fg.shape[0]
            _, num_true_rel_ind = torch.sort(perm_rel)
            num_true_rel_ind = num_true_rel_ind[:num_true_rel]

            result.num_true_rel = num_true_rel_ind
        else :
            result.gt_adj_mat_graph = gt_adj_mat

        if self.mode != 'sgdet':
            gt_mul_label_num = gt_mul_label_num
            result.gt_mul_label = gt_mul_label
            result.gt_mul_rel = gt_mul_rel
        else :
            gt_mul_label_num = result.gt_multi_label_num
            result.gt_mul_label = result.gt_mul_label
            result.gt_mul_rel = result.gt_mul_rel
        rel_inds_graph, rel_inds_offset_graph = self.get_rel_inds(result.rel_labels_graph, result.rel_labels_offset_graph, im_inds, boxes)

        rel_inds_rel, rel_inds_offset_rel = self.get_rel_inds(result.rel_labels_rel, result.rel_labels_offset_rel, im_inds, boxes)
        #rel_inds: shape [num_rels, 3], each array is [img_ind, box_ind1, box_ind2] for training
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)


        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # Prevent gradients from flowing back into score_fc from elsewhere
        if (self.training or self.mode == 'predcls') and self.mode != 'predcls_nongtbox':
            obj_labels = result.rm_obj_labels
        elif self.mode == 'predcls_nongtbox':
            rois, obj_labels, _, _, _, rel_labels = self.detector.gt_boxes(None, im_sizes, image_offset, gt_boxes,
                           gt_classes, gt_rels, train_anchor_inds, proposals=proposals)
            im_inds = rois[:, 0].long().contiguous()
            gt_boxes = rois[:, 1:]
            rel_inds = self.get_rel_inds(rel_labels, im_inds, gt_boxes)
            result.rel_labels = rel_labels
            result.rm_obj_labels = obj_labels

        else :
            obj_labels = None

        vr_rel = self.visual_rep(result.fmap.detach(), rois, rel_inds_rel[:, 1:])

        if result.gt_adj_mat_graph is not None:
            gt_adj_mat = result.gt_adj_mat_graph[rel_inds_graph[:, 1], \
                                                 rel_inds_offset_graph[:, 2]].type(torch.cuda.LongTensor).type_as(result.fmap.data)
        else:
            gt_adj_mat = None
        '''
        obj_dists2, obj_preds, edge_ctx
        '''
        result.rm_obj_dists, result.obj_preds_nozeros, edge_ctx = self.context(
            obj_fmaps = result.obj_fmap,
            obj_logits = result.rm_obj_dists.detach(),
            im_inds = im_inds,
            obj_labels = obj_labels,
            box_priors = boxes.data,
            boxes_per_cls = result.boxes_all,
        )
        result.im_inds = im_inds
        result.obj_feat_im_inds = obj_feat_im_inds

        if self.mode == 'detclass' :
            if self.training:
                return result
            else:
                rm_obj_softmax = F.softmax(result.rm_obj_dists, dim=-1)
                obj_preds = rm_obj_softmax[:, 1:].max(-1)[1] + 1
                twod_inds = arange(obj_preds.data) * self.num_classes + obj_preds.data
                obj_scores = F.softmax(result.rm_obj_dists, dim=-1)
                obj_scores = obj_scores.cpu().data.numpy()
                obj_preds = obj_preds.cpu().data.numpy()
                rm_box_priors = result.rm_box_priors.cpu().data.numpy()
                return rm_box_priors, obj_preds, obj_scores

        subj_rep = self.post_sub_lstm(edge_ctx)
        obj_rep = self.post_obj_lstm(edge_ctx)

        subj_rep = F.dropout(subj_rep, self.dropout_rate, training=self.training)
        obj_rep = F.dropout(obj_rep, self.dropout_rate, training=self.training)


        vr_obj = vr_rel
        subj_rep_rel = subj_rep[rel_inds_rel[:, 1]]
        obj_rep_rel = obj_rep[rel_inds_rel[:, 2]]
        #prod_rep = subj_rep[obj_rel_ind[:, 1]] * obj_rep[obj_rel_ind[:, 2]]
        if self.use_vision:

            if self.limit_vision:
                # exact value TBD
                subj_rep_rel = torch.cat((subj_rep_rel[:,:2048] * subj_rep_rel[:,:2048], subj_rep_rel[:,2048:]), 1)
                obj_rep_rel = torch.cat((obj_rep_rel[:, :2048] * obj_rep_rel[:, :2048], obj_rep_rel[:, 2048:]), 1)
            else:
                subj_rep_rel = subj_rep_rel * vr_obj
                obj_rep_rel = obj_rep_rel * vr_obj

        if self.use_tanh:
            subj_rep_rel = F.tanh(subj_rep_rel)
            obj_rep_rel = F.tanh(obj_rep_rel)
        if self.with_biliner_score:
            result.rel_dists = self.bilinear_score(subj_rep_rel, obj_rep_rel)
        else:
            prod_rep = subj_rep_rel * obj_rep_rel
            result.rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            if self.mode != 'sgdet':
                rel_obj_preds = result.obj_preds_nozeros.clone()
            else:

                rel_obj_preds = result.obj_preds_nozeros.clone()

            freq_bias_so = self.freq_bias.index_with_labels(torch.stack((
                rel_obj_preds[rel_inds_rel[:, 1]],
                rel_obj_preds[rel_inds_rel[:, 2]],
            ), 1))
            result.rel_dists = result.rel_dists + freq_bias_so
        if self.training:
            return result


        twod_inds = arange(result.obj_preds_nozeros.data) * self.num_classes + result.obj_preds_nozeros.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        elif self.mode == 'predcls_nongtbox':
            bboxes = gt_boxes
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_scores = F.softmax(result.rel_dists, dim=1)


        return filter_dets(ids, im_inds, bboxes, result.obj_scores,
                           result.obj_preds_nozeros,
                           rel_inds = rel_inds_rel[:, 1:],
                           pred_scores = rel_scores,
                           nongt_box=self.mode=='predcls_nongtbox',
                           with_adj_mat = None,
                           with_gt_adj_mat=self.with_gt_adj_mat,
                           gt_adj_mat=gt_adj_mat,
                           alpha=None,
                           feat=subj_rep_rel*obj_rep_rel,
                           ext_feat=self.ext_feat,
                           mode=self.mode)

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
