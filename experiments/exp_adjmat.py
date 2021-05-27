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
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

# from lib.decoder_rnn import DecoderRNN, lstm_factory, LockedDropout
from lib.lstm.decoder_rnn import DecoderRNN
# from lib.lstm.decoder_rnn_bg import DecoderRNN
from lib.lstm.highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM
from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg, load_resnet
from lib.pytorch_misc import transpose_packed_sequence_inds, to_onehot, arange, enumerate_by_image, diagonal_inds, \
    Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.gcn.pygcn import GraphConvolution
from lib.gcn.pygat import GraphAttentionLayer
from lib.lstm.mu_rnn import MultiLabelRNN
from lib.pytorch_misc import random_choose
import math
from lib.gcn.gat_model import GAT
from lib.sqrtm import sqrtm


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
                 embed_dim=200, hidden_dim=256, obj_dim=2048, pooling_dim=2048,
                 nl_mul=0, nl_obj=2, nl_obj_gcn=2, nl_edge=2, nl_adj=1, dropout_rate=0.2, order='confidence',
                 pass_in_obj_feats_to_decoder=True,
                 pass_in_obj_feats_to_edge=True,
                 pass_in_obj_feats_to_gcn=False,
                 pass_embed_togcn=False,
                 attention_dim=256, with_adj_mat=False,
                 max_num_obj=65, adj_embed_dim=256, mean_union_feat=False, adj_embed=False,
                 ch_res=False, use_bias=True, use_tanh=True,
                 limit_vision=True, use_vision=True,
                 bg_num_rel=-1, bg_num_graph=-1,
                 with_gcn=False, fb_thr=0.5, with_biliner_score=False,
                 gcn_adj_type='hard', num_gcn_layer=1, relu_alpha=0.2,
                 nhead=4, where_gcn=False, with_gt_adj_mat=False, type_gcn='normal',
                 edge_ctx_type='obj', nms_union=False):
        super(LinearizedContext, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        assert mode in MODES
        self.mode = mode
        self.attention_dim = attention_dim
        self.nl_mul = nl_mul
        self.nl_obj = nl_obj
        self.nl_obj_gcn = nl_obj_gcn
        self.nl_adj = nl_adj
        self.nl_edge = nl_edge
        self.with_adj_mat = with_adj_mat
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.pooling_dim = pooling_dim
        self.dropout_rate = dropout_rate
        self.pass_in_obj_feats_to_decoder = pass_in_obj_feats_to_decoder
        self.pass_in_obj_feats_to_edge = pass_in_obj_feats_to_edge
        self.pass_in_obj_feats_to_gcn = pass_in_obj_feats_to_gcn
        self.pass_embed_togcn = pass_embed_togcn
        self.max_num_obj = max_num_obj
        self.adj_embed_dim = adj_embed_dim
        self.mean_union_feat = mean_union_feat
        self.adj_embed = adj_embed
        self.ch_res = ch_res
        self.use_bias = use_bias
        self.use_vision = use_vision
        self.use_tanh = use_tanh
        self.limit_vision = limit_vision
        self.with_gcn = with_gcn
        self.fb_thr = fb_thr
        self.relu_alpha = relu_alpha
        self.with_biliner_score = with_biliner_score
        self.gcn_adj_type = gcn_adj_type
        self.num_gcn_layer = num_gcn_layer
        self.nhead = nhead
        self.where_gcn = where_gcn
        self.with_gt_adj_mat = with_gt_adj_mat
        self.type_gcn = type_gcn

        self.edge_ctx_type = edge_ctx_type
        self.nms_union = nms_union
        assert order in ('size', 'confidence', 'random', 'leftright')
        self.order = order
        if self.mode == 'predcls_nongtbox':
            self.order = 'random'
        # EMBEDDINGS
        # self.leakyrelu = nn.LeakyReLU(self.alpha)
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(self.num_classes, self.embed_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()
        if self.pass_embed_togcn:
            self.obj_embed3 = nn.Embedding(self.num_classes, self.embed_dim)
            self.obj_embed3.weight.data = embed_vecs.clone()
        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])
        self.pos_embed1 = nn.Sequential(*[
            nn.BatchNorm1d(4, momentum=BATCHNORM_MOMENTUM / 10.0),
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

        if self.nl_obj > 0 and self.mode != 'predcls_nongtbox':
            obj_ctx_rnn_indim = self.obj_dim + self.embed_dim + 128
            # obj_ctx_rnn_indim = self.obj_dim + self.embed_dim
            self.obj_ctx_rnn = AlternatingHighwayLSTM(
                input_size=obj_ctx_rnn_indim,
                hidden_size=self.hidden_dim,
                num_layers=self.nl_obj,
                recurrent_dropout_probability=dropout_rate)

            decoder_inputs_dim = self.hidden_dim
            if self.pass_in_obj_feats_to_decoder:
                decoder_inputs_dim += self.obj_dim + self.embed_dim + 128
            if self.with_gcn:
                decoder_inputs_dim = decoder_inputs_dim + self.hidden_dim
            self.decoder_rnn = DecoderRNN(self.classes, embed_dim=self.embed_dim,
                                          inputs_dim=decoder_inputs_dim,
                                          hidden_dim=self.hidden_dim,
                                          recurrent_dropout_probability=dropout_rate,
                                          mode=self.mode)

        if self.nl_obj == 0 or self.with_gcn:
            obj_gcn_input_dim = self.obj_dim + self.embed_dim + 128
            if self.where_gcn == 'stack':
                obj_gcn_input_dim = self.hidden_dim
            print('building obj gcn!')
            self.obj_gc1 = GraphConvolution(obj_gcn_input_dim, self.hidden_dim)
            # self.obj_gc1_re = gcn_layer(obj_gcn_input_dim, self.hidden_dim)
            # self.obj_gc2 = gcn_layer(self.hidden_dim, self.hidden_dim)
            # self.obj_gc2_re = gcn_layer(self.hidden_dim, self.hidden_dim)
            # self.obj_gc1_fb = gcn_layer(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
            # self.obj_gc_obj2linear = nn.Linear(obj_gcn_input_dim,self.hidden_dim)
            # self.obj_gc_obj2linear.weight = torch.nn.init.xavier_normal(self.obj_gc_obj2linear.weight, gain=1.0)
            if self.nl_obj == 0:
                self.decoder_lin = nn.Linear(self.hidden_dim, self.num_classes)
                self.decoder_lin.weight = torch.nn.init.xavier_normal(self.decoder_lin.weight, gain=1.0)

        if self.with_adj_mat:
            if self.nl_adj > 0:
                adj_input_dim = self.obj_dim + self.embed_dim + 128 + int(self.obj_dim / 2)
                if ch_res:
                    adj_input_dim = adj_input_dim + self.hidden_dim
                self.adj_mat_rnn = AlternatingHighwayLSTM(input_size=adj_input_dim,
                                                          hidden_size=self.hidden_dim,
                                                          num_layers=self.nl_adj,
                                                          recurrent_dropout_probability=dropout_rate)
                self.adj_mat_embed_decoder = nn.Linear(self.hidden_dim, self.adj_embed_dim)
                self.adj_mat_lin = nn.Linear(self.adj_embed_dim, 2)
                self.adj_mat_embed_encoder = nn.Linear(self.max_num_obj, self.adj_embed_dim)
                self.feat_adj_mat_lin = nn.Linear(self.obj_dim, self.adj_embed_dim)

        if (self.with_adj_mat and not self.with_gt_adj_mat) or self.type_gcn == 'gat':
            if self.nl_obj > 0:
                post_lstm_in = self.hidden_dim
                if self.with_gcn and self.where_gcn == 'parall':
                    post_lstm_in = post_lstm_in + self.hidden_dim
                elif self.with_gcn and self.where_gcn == 'stack':
                    post_lstm_in = self.hidden_dim
                self.post_lstm_graph_obj = nn.Linear(post_lstm_in, self.pooling_dim)
                self.post_lstm_graph_sub = nn.Linear(post_lstm_in, self.pooling_dim)
                '''
                self.post_lstm_graph_obj.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
                self.post_lstm_graph_obj.bias.data.zero_()
                self.post_lstm_graph_sub.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
                self.post_lstm_graph_sub.bias.data.zero_()
                '''
                self.post_lstm_graph_obj.weight = torch.nn.init.xavier_normal(self.post_lstm_graph_obj.weight, gain=1.0)
                self.post_lstm_graph_sub.weight = torch.nn.init.xavier_normal(self.post_lstm_graph_sub.weight, gain=1.0)
                post_lstm_dim = self.hidden_dim
                if self.with_gcn and self.where_gcn == 'parall':
                    post_lstm_dim = self.pooling_dim + self.embed_dim + 128
                elif self.with_gcn and self.where_gcn == 'stack':
                    post_lstm_dim = self.hidden_dim

                self.post_lstm_graph_obj_1 = nn.Linear(post_lstm_dim, self.pooling_dim)
                self.post_lstm_graph_sub_1 = nn.Linear(post_lstm_dim, self.pooling_dim)
                '''
                self.post_lstm_graph_obj_1.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
                self.post_lstm_graph_obj_1.bias.data.zero_()
                self.post_lstm_graph_sub_1.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
                self.post_lstm_graph_sub_1.bias.data.zero_()
                '''
                self.post_lstm_graph_obj_1.weight = torch.nn.init.xavier_normal(self.post_lstm_graph_obj_1.weight,
                                                                                gain=1.0)
                self.post_lstm_graph_sub_1.weight = torch.nn.init.xavier_normal(self.post_lstm_graph_sub_1.weight,
                                                                                gain=1.0)
            if self.nl_obj == 0:
                self.post_lstm_graph = nn.Linear(self.pooling_dim + self.embed_dim + 128, self.pooling_dim * 2)
                # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1.
                # (Half contribution comes from LSTM, half from embedding.
                # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
                self.post_lstm_graph.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.hidden_dim))
                self.post_lstm_graph.bias.data.zero_()
            if self.with_biliner_score:
                self.graph_bilinear = nn.Linear(self.pooling_dim, 2, bias=False)
                self.graph_bilinear.weight = torch.nn.init.xavier_normal(self.graph_bilinear.weight, gain=1.0)
                self.graph_bilinear_1 = nn.Linear(self.pooling_dim, 2, bias=False)
                self.graph_bilinear_1.weight = torch.nn.init.xavier_normal(self.graph_bilinear_1.weight, gain=1.0)
            else:
                self.rel_compress_graph = nn.Linear(self.pooling_dim, 2, bias=True)
                self.rel_compress_graph.weight = torch.nn.init.xavier_normal(self.rel_compress_graph.weight, gain=1.0)
            if self.use_bias:
                self.freq_bias_graph = FrequencyBias(graph=True, with_bg=bg_num_graph != 0)
        if self.mode == 'detclass':
            return
        if self.nl_edge > 0:
            input_dim = self.embed_dim + self.obj_dim + 128
            # input_dim = self.embed_dim + self.obj_dim
            if self.nl_obj > 0 and self.mode != 'predcls_nongtbox':
                input_dim += 0
            else:
                input_dim += self.obj_dim + self.embed_dim + 128
            if self.ch_res:
                input_dim += self.hidden_dim
            if self.with_gcn:
                input_dim += 0
            if self.edge_ctx_type == 'union':
                input_dim = 2 * self.embed_dim + self.pooling_dim
            # if self.with_adj_mat:
            #    input_dim += self.adj_embed_dim
            if self.mode == 'predcls_nongtbox':
                self.obj_feat_att = nn.Linear(self.obj_dim + self.embed_dim + 128, self.attention_dim)
                self.obj_label_att = nn.Linear(self.embed_dim, self.attention_dim)
                # self.obj_feat_label_att = nn.Linear( self.obj_dim + self.embed_dim + 128+self.embed_dim, self.attention_dim)
                self.att_weight = nn.Linear(self.attention_dim, 1)
            # print('input_dim: ',input_dim)
            self.edge_ctx_rnn = AlternatingHighwayLSTM(input_size=input_dim,
                                                       hidden_size=self.hidden_dim,
                                                       num_layers=self.nl_edge,
                                                       recurrent_dropout_probability=dropout_rate)
            if self.edge_ctx_type == 'union':
                decoder_inputs_dim = 3 * self.hidden_dim

                self.decoder_edge_rnn = DecoderRNN(self.rel_classes, embed_dim=self.embed_dim,
                                                   inputs_dim=decoder_inputs_dim,
                                                   hidden_dim=self.hidden_dim,
                                                   recurrent_dropout_probability=dropout_rate,
                                                   type=self.edge_ctx_type)
                '''
                self.decoder_edge_linear = nn.Linear(decoder_inputs_dim, self.num_rels, bias=True)
                self.decoder_edge_linear.weight = torch.nn.init.xavier_normal(self.decoder_edge_linear.weight, gain=1.0)
                '''
        if self.with_gcn:
            print('building gcn! number layers', self.num_gcn_layer)
            gcn_input_dim = self.hidden_dim
            if self.where_gcn == 'parall':
                gcn_input_dim += self.hidden_dim
            if self.pass_embed_togcn:
                gcn_input_dim += self.embed_dim
            if self.pass_in_obj_feats_to_gcn:
                gcn_input_dim += self.obj_dim + self.embed_dim + 128

            self.gc_list = []
            self.gc_re_list = []

            self.gc2_list = []
            self.gc2_re_list = []
            ''' '''
            for i in range(self.num_gcn_layer):
                self.gc_list.append(GraphConvolution(gcn_input_dim, self.hidden_dim))
                # self.gc_re_list.append(gcn_layer(gcn_input_dim, self.hidden_dim))
                '''
                self.gc2_list.append(gcn_layer(hidden_dim, self.hidden_dim))
                self.gc2_re_list.append(gcn_layer(hidden_dim, self.hidden_dim))
                '''
            self.gc = nn.Sequential(*self.gc_list)
            # self.gc_re = nn.Sequential(*self.gc_re_list)
            '''
            self.gc2 = nn.Sequential(*self.gc2_list)
            self.gc2_re = nn.Sequential(*self.gc2_re_list)
            '''
            self.gcn_input_dim = gcn_input_dim

    def sort_rois(self, batch_idx, confidence, box_priors):
        """
        :param batch_idx: tensor with what index we're on
        :param confidence: tensor with confidences between [0,1)
        :param boxes: tensor with (x1, y1, x2, y2)
        :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
        """
        cxcywh = center_size(box_priors)
        if self.order == 'size':
            sizes = cxcywh[:, 2] * cxcywh[:, 3]
            # sizes = (box_priors[:, 2] - box_priors[:, 0] + 1) * (box_priors[:, 3] - box_priors[:, 1] + 1)
            assert sizes.min() > 0.0
            scores = sizes / (sizes.max() + 1)
        elif self.order == 'confidence':
            scores = confidence
        elif self.order == 'random':
            scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
        elif self.order == 'leftright':
            centers = cxcywh[:, 0]
            scores = centers / (centers.max() + 1)
        else:
            raise ValueError("invalid mode {}".format(self.order))
        return _sort_by_score(batch_idx, scores)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def convert_symmat(self, adj):
        adj_union = (adj.permute(1, 0) > adj).type_as(adj)
        adj = adj + (adj.permute(1, 0)).mul(adj_union) - adj.mul(adj_union)
        return adj

    def edge_ctx(self, obj_feats, obj_dists, im_inds, obj_preds, pos_embed, box_priors=None):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_embed2 = self.obj_embed2(obj_preds)

        inp_feats = torch.cat((obj_feats, obj_embed2, pos_embed), 1)
        # inp_feats = torch.cat((obj_feats, obj_embed2),1)
        # Sort by the confidence of the maximum detection.
        confidence = F.softmax(obj_dists, dim=1).data.view(-1)[
            obj_preds.data + arange(obj_preds.data) * self.num_classes]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, confidence, box_priors)
        # print('inp_feats: ',inp_feats.size())
        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0][0]

        # now we're good! unperm
        edge_ctx = edge_reps[inv_perm]
        return edge_ctx

    def obj_ctx(self, obj_feats, obj_dists, im_inds, obj_labels=None, box_priors=None, boxes_per_cls=None,
                obj_ctx_gcn=None, rel_inds=None, rel_inds_offset=None,
                union_feat=None, label_prior=None, gt_adj_mat=None):
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
        confidence = F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[0]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, confidence, box_priors)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()

        input_packed = PackedSequence(obj_inp_rep, ls_transposed)

        encoder_rep = self.obj_ctx_rnn(input_packed)[0][0]
        # Decode in order
        adj_dis_softmax_obj = None
        if self.with_gcn and self.where_gcn == 'parall':
            obj_ctx_gcn = obj_ctx_gcn[perm].contiguous()
            encoder_rep = torch.cat([obj_ctx_gcn, encoder_rep], -1)
        adj_dist_owbias_obj = None
        if self.with_gcn and self.where_gcn == 'stack':
            encoder_rep_r = encoder_rep[inv_perm]
            adj_pre = encoder_rep_r
            if not self.with_gt_adj_mat:
                if self.mode == 'sgdet':
                    # adj_obj_preds = obj_dists.max(1)[1]
                    '''
                    if self.training:
                        #print('obj_labels', obj_labels)
                        adj_obj_preds = obj_labels.clone()
                        nonzero_pred = obj_dists[:, 1:].max(1)[1] + 1
                        is_bg = (adj_obj_preds.data == 0).nonzero()
                        if is_bg.dim() > 0:
                            adj_obj_preds[is_bg.squeeze(1)] = nonzero_pred[is_bg.squeeze(1)]
                        #print('obj_preds',obj_preds)
                    else:
                        adj_obj_preds = label_prior
                        #print('obj_preds', obj_preds)
                    '''

                    if self.training:
                        adj_obj_preds = obj_labels.clone()
                    else:
                        adj_obj_preds = obj_dists.max(1)[1]
                else:
                    if self.training or self.mode == 'predcls':
                        adj_obj_preds = obj_labels
                    else:
                        adj_obj_preds = obj_dists[:, 1:].max(1)[1] + 1
                adj_dist_owbias_obj = self.from_pre_to_mat(obj_pre=adj_pre, sub_pre=adj_pre, rel_inds_graph=rel_inds,
                                                           vr_graph=union_feat,
                                                           rel_inds_offset_graph=rel_inds_offset,
                                                           obj_preds=adj_obj_preds, num=1)

                adj_dis_softmax_obj = F.softmax(adj_dist_owbias_obj, -1)[:, 1]
            else:
                adj_dis_softmax_obj = gt_adj_mat
            obj_ctx_gcn \
                = self.obj_gcn(encoder_rep_r,
                               im_inds=im_inds,
                               rel_inds=rel_inds,
                               adj_mat=adj_dis_softmax_obj,  # if not self.training else gt_adj_mat,
                               )
            obj_ctx_gcn = obj_ctx_gcn[perm].contiguous()
            encoder_rep = torch.cat([obj_ctx_gcn, encoder_rep], -1)
        decoder_inp = PackedSequence(
            torch.cat((obj_inp_rep, encoder_rep), 1) if self.pass_in_obj_feats_to_decoder else encoder_rep,
            ls_transposed)
        # print(decoder_inp.size())
        # print('obj_labels: ',obj_labels)
        obj_dists, obj_preds_nozeros, obj_preds_zeros, decoder_rep = self.decoder_rnn(
            decoder_inp,  # obj_dists[perm],
            labels=obj_labels[perm] if obj_labels is not None else None,
            boxes_for_nms=boxes_per_cls[perm] if boxes_per_cls is not None else None,
            obj_dists=obj_dists[perm],
        )

        obj_preds_nozeros = obj_preds_nozeros[inv_perm]
        obj_preds_zeros = obj_preds_zeros[inv_perm]
        # print('obj_preds: ', obj_preds)
        obj_dists = obj_dists[inv_perm]
        decoder_rep = decoder_rep[inv_perm]
        if self.mode == 'predcls':
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = Variable(to_onehot(obj_preds.data, self.num_classes))
            decoder_rep = decoder_rep
            obj_preds_nozeros = obj_preds
            obj_preds_zeros = obj_preds
        encoder_rep = encoder_rep[inv_perm]

        return obj_dists, obj_preds_nozeros, obj_preds_zeros, encoder_rep, decoder_rep, adj_dist_owbias_obj

    def obj_gcn(self, obj_feats, im_inds, rel_inds, adj_mat, num_layer=0,
                obj_labels=None, box_priors=None, boxes_per_cls=None):

        if self.gcn_adj_type == 'hard':
            adj_mat_t = adj_mat > self.fb_thr + 0.01
            adj_mat_t = adj_mat_t.type_as(obj_feats)
        if self.gcn_adj_type == 'soft':
            adj_mat_t = adj_mat
        spare_adj_mat = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(obj_feats.get_device(), async=True)
        spare_adj_mat = Variable(spare_adj_mat)
        spare_adj_mat[rel_inds[:, 1], rel_inds[:, 2]] = adj_mat_t
        spare_adj_mat = self.convert_symmat(spare_adj_mat)
        spare_adj_mat = self.adj_to_Laplacian(spare_adj_mat, type=self.gcn_adj_type)
        x = F.elu(self.obj_gc1(obj_feats, spare_adj_mat))
        x = F.dropout(x, self.dropout_rate, training=self.training)

        '''
        spare_adj_mat_re = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(obj_feats.get_device(), async=True)
        spare_adj_mat_re = Variable(spare_adj_mat_re)
        spare_adj_mat_re[rel_inds[:, 2], rel_inds[:, 1]] = adj_mat_t
        spare_adj_mat_re = self.adj_to_Laplacian(spare_adj_mat_re, type=self.gcn_adj_type)
        x_re = F.relu(self.obj_gc1_re(obj_feats, spare_adj_mat_re))
        x_re = F.dropout(x_re, self.dropout_rate, training=self.training)
        '''

        '''
        x = F.relu(self.obj_gc2(x, spare_adj_mat))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x_re = F.relu(self.obj_gc2_re(x_re, spare_adj_mat_re))
        x_re = F.dropout(x_re, self.dropout_rate, training=self.training)
        '''
        return x

    def gen_adj_pre(self, obj_feats, obj_dists, im_inds, box_priors=None, use_bias=True):
        adj_mat = None
        '''
        offset_num = []
        num_imgs = rel_inds[:,0]
        max_num_imgs = num_imgs.cpu().numpy().max()
        max_num_imgs = int(max_num_imgs) + 1
        rel_img_inds = num_imgs
        img_id = torch.arange(max_num_imgs)
        offset_id = torch.zeros_like(img_id)
        id_ind_mat = im_inds[:,None] == im_inds[None,:]
        rel_ind_mat = rel_img_inds[:,None] == rel_img_inds[None,:]
        id_ind_mat = id_ind_mat.type_as(img_id)
        num_per_imgs = id_ind_mat.sum(0)
        offset_id[1:] = num_per_imgs[:-2]
        offset_ind = rel_ind_mat * offset_id[None,:]
        offset_ind = offset_ind.sum(1)

        imd_ind_mat = im_inds[:,None] == im_inds[None, :]
        imd_ind_mat = imd_ind_mat.type_as(obj_feats)

        obj_rep = self.adj_mat_embed_decoder(obj_rep)
        obj_rep = F.relu(obj_rep,True)
        obj_rep_union = obj_rep[:,None,:] * obj_rep[None,:,:] * imd_ind_mat[:,:,None]
        feat_adj_union = self.feat_adj_mat_lin(union_feat)
        feat_adj_union = F.relu(feat_adj_union,True)
        feat_adj_union_mat = torch.zeros_like(obj_rep_union)
        feat_adj_union_mat[rel_inds[:,1],rel_inds[:,2],:] = feat_adj_union
        feat_adj_union_mat = imd_ind_mat[:,:,None] * feat_adj_union_mat
        adj_rep_union = obj_rep_union + feat_adj_union_mat

        adj_mat_batch_id = self.adj_mat_lin(adj_rep_union)
        #adj_mat_batch_id = F.softmax(adj_mat_batch_id, -1)
        adj_mat_t = torch.zeros([adj_mat_batch_id.size(0),self.max_num_obj,2]).cuda(adj_mat_batch_id.get_device(),async=True)
        adj_mat = torch.autograd.Variable(adj_mat_t)
        adj_mat[rel_inds[:,1],rel_inds_offset[:,2]] = adj_mat_batch_id[rel_inds[:,1],rel_inds[:,2],]
        '''
        # adj_mat = F.sigmoid(adj_mat)
        confidence = F.softmax(obj_dists, dim=1).data[:, 1:].max(1)[0]
        perm, inv_perm, ls_transposed = self.sort_rois(im_inds.data, confidence, box_priors)
        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)

        obj_rep = self.adj_mat_rnn(input_packed)[0][0]
        adj_mat = obj_rep[inv_perm]
        return adj_mat

    def from_pre_to_mat(self, obj_pre, sub_pre, rel_inds_graph, vr_graph, rel_inds_offset_graph, \
                        obj_preds, num=0):
        if num == 0:
            subj_rep_graph = self.post_lstm_graph_sub(sub_pre)
            obj_rep_graph = self.post_lstm_graph_obj(obj_pre)

        if num == 1:
            subj_rep_graph = self.post_lstm_graph_sub_1(sub_pre)
            obj_rep_graph = self.post_lstm_graph_obj_1(obj_pre)

        subj_rep_graph = F.dropout(subj_rep_graph, self.dropout_rate, training=self.training)
        obj_rep_graph = F.dropout(obj_rep_graph, self.dropout_rate, training=self.training)
        obj_rel_ind_graph = rel_inds_graph
        vr_obj_graph = vr_graph
        subj_rep_graph_rel = subj_rep_graph[rel_inds_graph[:, 1]]
        obj_rep_graph_rel = obj_rep_graph[rel_inds_graph[:, 2]]
        # prod_rep_graph = subj_rep_graph_rel * obj_rep_graph_rel
        if self.use_vision:
            if self.mode != 'predcls_nongtbox':
                # prod_rep_graph = prod_rep_graph
                subj_rep_graph_rel = subj_rep_graph_rel
                obj_rep_graph_rel = obj_rep_graph_rel
            if self.limit_vision:
                # exact value TBD
                subj_rep_graph_rel = torch.cat((subj_rep_graph_rel[:, :self.pooling_dim] * vr_obj_graph[:,
                                                                                           :self.pooling_dim],
                                                subj_rep_graph_rel[:, self.pooling_dim:]), 1)
                obj_rep_graph_rel = torch.cat((obj_rep_graph_rel[:, :self.pooling_dim] * vr_obj_graph[:,
                                                                                         :self.pooling_dim],
                                               obj_rep_graph_rel[:, self.pooling_dim:]), 1)
            else:
                subj_rep_graph_rel = subj_rep_graph_rel * vr_obj_graph[:, :self.pooling_dim]
                obj_rep_graph_rel = obj_rep_graph_rel * vr_obj_graph[:, :self.pooling_dim]

        if self.use_tanh:
            # prod_rep_graph = F.tanh(prod_rep_graph)
            subj_rep_graph_rel = F.tanh(subj_rep_graph_rel)
            obj_rep_graph_rel = F.tanh(obj_rep_graph_rel)
        if self.with_biliner_score:
            rel_dists_graph = self.bilinear_score_graph(subj_rep_graph_rel, obj_rep_graph_rel, num)
        else:
            prod_rep_graph = subj_rep_graph_rel * obj_rep_graph_rel
            rel_dists_graph = self.rel_compress_graph(prod_rep_graph)

        if self.use_bias:
            rel_dists_graph = self.freq_bias_graph.index_with_labels(torch.stack((
                obj_preds[obj_rel_ind_graph[:, 1]],
                obj_preds[obj_rel_ind_graph[:, 2]],
            ), 1))
            '''
            rel_dists_graph = rel_dists_graph + self.freq_bias_graph.index_with_labels(torch.stack((
                obj_preds[obj_rel_ind_graph[:, 1]],
                obj_preds[obj_rel_ind_graph[:, 2]],
            ), 1))
            '''
            '''
            rel_dists_graph = rel_dists_graph + F.dropout(self.freq_bias_graph.index_with_labels(torch.stack((
                obj_preds[obj_rel_ind_graph[:, 1]],
                obj_preds[obj_rel_ind_graph[:, 2]],
            ), 1)),self.dropout_rate, training=self.training)
            '''
            rel_dists_graph = rel_dists_graph
            # adj_mat_t = torch.zeros([adj_pre.size(0),self.max_num_obj,2]).cuda(rel_dists_graph.get_device(),async=True)
            # pre_adj_mat = Variable(adj_mat_t)
            # pre_adj_mat[rel_inds_graph[:,1],rel_inds_offset_graph[:,2]] = rel_dists_graph

        return rel_dists_graph

    def adj_to_Laplacian_spare(self, adj_mat):
        eye_mat = torch.ones([1, adj_mat.size(0)])
        eye_coo = torch.arange(adj_mat.size(0))
        eye_coo = torch.cat([eye_coo[:, None], eye_coo[:, None]], -1)
        spare_eye_mat = torch.sparse.FloatTensor(eye_coo, eye_mat, \
                                                 torch.Size([adj_mat.size(0), adj_mat.size(0)])).cuda(
            adj_mat.get_device(), async=True)

        adj_mat = adj_mat + spare_eye_mat
        '''
        degree_mat = torch.dot(adj_mat, adj_mat_t) * eye_mat
        degree_mat_re = 1 / (degree_mat + 1e-8) * eye_mat
        degree_mat_re = degree_mat_re ** (0.5)
        '''
        degree_mat = adj_mat.sum(-1)
        degree_mat_re = 1 / (degree_mat + 1e-8)
        dot_1 = degree_mat_re[:, None] * adj_mat
        # print(dot_1.size(),degree_mat_re.size(),adj_mat.size())
        return dot_1

    def adj_to_Laplacian(self, adj_mat, type='hard'):
        eye_mat = torch.eye(adj_mat.size(0)).cuda(adj_mat.get_device(), async=True)
        eye_mat = Variable(eye_mat)
        adj_mat = adj_mat + eye_mat
        '''
        degree_vec = adj_mat.sum(-1)
        #degree_vec = torch.sqrt(degree_vec)
        degree_mat_re_vec = (1 / (degree_vec + 1e-8))

        degree_vec_r = adj_mat.sum(0)
        #degree_vec_r = torch.sqrt(degree_vec_r)
        degree_mat_re_vec_r = (1 / (degree_vec_r + 1e-8))

        degree_mat_re = torch.zeros_like(adj_mat)
        degree_mat_re_r = torch.zeros_like(adj_mat)
        i = np.arange(adj_mat.size(0))
        degree_mat_re[i,i] = degree_mat_re_vec
        degree_mat_re_r[i,i] = degree_mat_re_vec_r
        #adj_mat_L = degree_mat_re - adj_mat

        dot_1 = (degree_mat_re @ adj_mat) @ degree_mat_re_r
        #dot_1 = degree_mat_re @ adj_mat
        '''
        degree_mat = adj_mat.sum(-1)
        degree_mat_re = 1 / (degree_mat + 1e-8)
        dot_1 = degree_mat_re[:, None] * adj_mat

        # print(dot_1.size(),degree_mat_re.size(),adj_mat.size())
        return dot_1

    def gcn_ctx(self, ctx_feat, adj_mat, im_inds, rel_inds, num_layer=0, obj_preds=None,
                rel_inds_offset=None):
        '''
        Args:
            ctx_feat:
            adj_mat:
            im_inds:
            rel_inds:
            rel_inds_offset:

        Returns:

        '''

        '''
        if adj_mat is None:
            spare_value = torch.ones([rel_inds.size(0),rel_inds.size(0)]).cuda(ctx_feat.get_device(),async=True)
            spare_value = Variable(spare_value)
        else:
            spare_value = adj_mat > self.fb_thr
            spare_value = spare_value.type_as(ctx_feat)
            spare_value = spare_value[rel_inds[:,1],rel_inds_offset[:,2]]
            #spare_value = spare_value
        spare_adj_mat = torch.sparse.FloatTensor(rel_inds[1:3],spare_value, \
                                                     torch.Size([im_inds.size(0),im_inds.size(0)]))

        spare_adj_mat_re = torch.sparse.FloatTensor(rel_inds[1:3][:,::-1], spare_value, \
                                                     torch.Size([im_inds.size(0), im_inds.size(0)]))
        coo_rel = torch.where(adj_mat > self.fb_thr + 0.01)
        ones_rel = torch.ones([1, coo_rel.size(0)]).cuda(ctx_feat.get_device(),async=True)
        spare_adj_mat = torch.sparse.FloatTensor(coo_rel[0:2], ones_rel, \
                                                     torch.Size([im_inds.size(0),im_inds.size(0)])).cuda(ctx_feat.get_device(),async=True)
        spare_adj_mat_re = torch.sparse.FloatTensor(coo_rel[0:2][:, ::-1], ones_rel, \
                                                    torch.Size([im_inds.size(0), im_inds.size(0)])).cuda(ctx_feat.get_device(),async=True)

        adj_mat_t = adj_mat > self.fb_thr + 0.01
        adj_mat_t = adj_mat_t.type_as(ctx_feat)
        spare_adj_mat = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(ctx_feat.get_device(),async=True)
        spare_adj_mat_re = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(ctx_feat.get_device(),async=True)
        spare_adj_mat = Variable(spare_adj_mat)
        spare_adj_mat_re = Variable(spare_adj_mat_re)
        spare_adj_mat[rel_inds[:,1],rel_inds[:,2]] = adj_mat_t[rel_inds[:,1],rel_inds[:,2]]
        spare_adj_mat_re[rel_inds[:, 2], rel_inds[:, 1]] = adj_mat_t[rel_inds[:, 1], rel_inds[:, 2]]

        spare_adj_mat = self.adj_to_Laplacian(spare_adj_mat)
        spare_adj_mat_re = self.adj_to_Laplacian(spare_adj_mat_re)
        '''
        if self.gcn_adj_type == 'hard':
            adj_mat_t = adj_mat > self.fb_thr + 0.01
            adj_mat_t = adj_mat_t.type_as(ctx_feat)
        if self.gcn_adj_type == 'soft':
            adj_mat_t = adj_mat
        # pred_embed = self.obj_embed2(obj_preds)

        spare_adj_mat = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(ctx_feat.get_device(), async=True)
        spare_adj_mat = Variable(spare_adj_mat)
        spare_adj_mat[rel_inds[:, 1], rel_inds[:, 2]] = adj_mat_t
        spare_adj_mat = self.convert_symmat(spare_adj_mat)
        spare_adj_mat = self.adj_to_Laplacian(spare_adj_mat, type=self.gcn_adj_type)

        x = F.elu(self.gc[num_layer](ctx_feat, spare_adj_mat))
        x = F.dropout(x, self.dropout_rate, training=self.training)

        '''
        spare_adj_mat_re = torch.zeros([im_inds.size(0), im_inds.size(0)]).cuda(ctx_feat.get_device(),async=True)
        spare_adj_mat_re = Variable(spare_adj_mat_re)
        spare_adj_mat_re[rel_inds[:, 2], rel_inds[:, 1]] = adj_mat_t
        spare_adj_mat_re = self.adj_to_Laplacian(spare_adj_mat_re,type=self.gcn_adj_type)

        x_re = F.relu(self.gc_re[num_layer](ctx_feat, spare_adj_mat_re))
        x_re = F.dropout(x_re, self.dropout_rate, training=self.training)
        '''

        '''
        x = F.relu(self.gc2[num_layer](x, spare_adj_mat))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x_re = F.relu(self.gc2_re[num_layer](x_re, spare_adj_mat_re))
        x_re = F.dropout(x_re, self.dropout_rate, training=self.training)
        '''
        return x, x

    def bilinear_score_graph(self, sub, obj, num=0):
        '''

        Args:
            obj:  num_rel, dim_hid
            sub: num_rel, dim_hid

        Returns:

        '''
        prod_rep_graph = sub * obj
        if num == 0:
            rel_dists_graph = self.graph_bilinear(prod_rep_graph)
            # rel_dists_graph = F.dropout(rel_dists_graph, self.dropout_rate, training=self.training)
        if num == 1:
            rel_dists_graph = self.graph_bilinear_1(prod_rep_graph)
            # rel_dists_graph = F.dropout(rel_dists_graph, self.dropout_rate, training=self.training)
        return rel_dists_graph

    def bilinear_score_graph_obj(self, sub, obj):
        '''

        Args:
            obj:  num_rel, dim_hid
            sub: num_rel, dim_hid

        Returns:

        '''
        prod_rep_graph = sub * obj
        rel_dists_graph = self.obj_graph_bilinear(prod_rep_graph)
        # rel_dists_graph = F.dropout(rel_dists_graph, self.dropout_rate, training=self.training)
        return rel_dists_graph

    def mul_ctx_T(self, obj_feats, im_inds, num_obj_per):
        '''
        num_obj_per = im_inds[:,None] == im_inds[None,:]
        num_obj_per = num_obj_per.type_as(im_inds)
        num_obj_per = num_obj_per.sum(-1)
        '''
        mul_dist, _, mul_state = self.mul_rnn(obj_feats, obj_num=num_obj_per, im_inds=im_inds)

        return mul_dist, mul_state

    def get_union_box(self, rois, union_inds):
        im_inds = rois[:, 0][union_inds[:, 0]]

        union_rois = torch.cat((
            im_inds[:, None],
            torch.min(rois[:, 1:3][union_inds[:, 0]], rois[:, 1:3][union_inds[:, 1]]),
            torch.max(rois[:, 3:5][union_inds[:, 0]], rois[:, 3:5][union_inds[:, 1]]),
        ), 1)
        return union_rois

    def forward(self, obj_fmaps, obj_logits, im_inds, obj_labels=None,
                box_priors=None, boxes_per_cls=None, obj_feat_im_inds=None, f_map=None, union_feat=None,
                rel_inds=None, rel_inds_offset=None, num_box=None, num_obj_per=None,
                gt_adj_mat=None, rel_label=None, label_prior=None):
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
        # if self.mode == 'predcls':
        #   obj_logits = Variable(to_onehot(obj_labels.data, self.num_classes))
        #   obj_logits_softmax = obj_logits

        obj_embed = obj_logits_softmax @ self.obj_embed.weight
        obj_embed = F.dropout(obj_embed, self.dropout_rate, training=self.training)
        pos_embed = self.pos_embed(Variable(center_size(box_priors)))
        pos_embed1 = self.pos_embed1(Variable(center_size(box_priors)))
        obj_pre_rep = torch.cat((obj_fmaps, obj_embed, pos_embed), 1)
        # obj_pre_rep = torch.cat((obj_fmaps, obj_embed), 1)
        adj_dist_owbias_rel = None
        adj_dist_owbias_obj = None
        obj_ctx_gcn = None

        if self.nl_obj > 0 and self.mode != 'predcls_nongtbox':

            obj_dists2, obj_preds_nozeros, obj_preds_zeros, \
            obj_ctx, decoder_rep, adj_dist_owbias_obj_t = self.obj_ctx(
                obj_pre_rep,
                obj_logits,
                im_inds,
                obj_labels,
                box_priors,
                boxes_per_cls,
                obj_ctx_gcn=obj_ctx_gcn,
                rel_inds=rel_inds,
                union_feat=union_feat,
                rel_inds_offset=rel_inds_offset,
                label_prior=label_prior,
                gt_adj_mat=gt_adj_mat,
            )
            if adj_dist_owbias_obj_t is not None:
                adj_dist_owbias_obj = adj_dist_owbias_obj_t
            if self.ch_res:
                obj_ctx = torch.cat([obj_ctx, decoder_rep], -1)

        elif self.with_gcn:

            obj_dists2 = self.decoder_lin(obj_ctx_gcn)
            obj_preds = obj_dists2[:, 1:].max(-1)[1] + 1
            obj_preds_nozeros = obj_preds
            obj_preds_zeros = obj_dists2.max(-1)[1]

        else:
            # UNSURE WHAT TO DO HERE
            if self.mode == 'predcls' or self.mode == 'predcls_nongtbox':
                obj_dists2 = Variable(to_onehot(obj_labels.data, self.num_classes))

            if (self.mode == 'sgdet') and not self.training:
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

                obj_preds_nozeros = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
                obj_preds_zeros = Variable(nms_mask * probs.data, volatile=True)[:, :].max(1)[1]


            else:
                obj_preds_nozeros = obj_labels if obj_labels is not None else obj_dists2[:, 1:].max(1)[1] + 1
                obj_preds_zeros = obj_labels if obj_labels is not None else obj_dists2[:, :].max(1)[1]

        if self.mode == 'detclass':
            return obj_dists2, obj_preds_nozeros, obj_preds_zeros, None, None, None, \
                   None, None, adj_dist_owbias_obj, None, None, None, None

        edge_ctx = None
        obj_feat_att = None
        obj_feat_att_w = None
        rel_inds_nms = rel_inds
        keep_union = None
        rel_dists = None
        if self.mode == 'sgdet':

            edge_obj_pred = obj_preds_nozeros.clone()
            '''
            if self.training :
                edge_obj_pred = obj_labels.clone()
            else:
                edge_obj_pred = obj_dists2.detach().max(1)[1]
            '''
        else:
            edge_obj_pred = obj_preds_nozeros.clone()
        if self.nl_edge > 0:
            edge_ctx = self.edge_ctx(
                obj_fmaps,
                obj_dists=obj_dists2.detach(),  # Was previously obj_logits.
                im_inds=im_inds,
                obj_preds=edge_obj_pred,  # obj_preds_zeros, #obj_preds_zeros obj_preds_nozeros
                box_priors=box_priors,
                pos_embed=pos_embed1,
            )
        elif self.nl_edge == 0:
            edge_ctx = obj_ctx

        edge_sub_ctx = None
        edge_obj_ctx = None

        if self.with_gcn:

            # gtc_input = torch.cat([edge_ctx,f_mean_map],-1)
            gtc_input = obj_ctx
            if self.where_gcn == 'stack':
                if self.with_adj_mat:
                    gtc_input = edge_ctx
                    adj_pre = edge_ctx
                    if self.mode == 'sgdet':
                        '''
                        adj_obj_preds = obj_preds_nozeros.clone()
                        '''
                        if self.training:
                            adj_obj_preds = obj_labels.clone()
                            # adj_obj_preds = obj_preds_nozeros.clone()
                        else:
                            adj_obj_preds = obj_dists2.detach().max(1)[1]
                            # adj_obj_preds = obj_preds_zeros
                            # adj_obj_preds = obj_preds_nozeros.clone()

                    else:
                        adj_obj_preds = obj_preds_nozeros.clone()
                    if not self.with_gt_adj_mat:
                        # print('obj_preds_nozeros',obj_preds_nozeros)
                        adj_dist_owbias_rel = self.from_pre_to_mat(obj_pre=adj_pre, sub_pre=adj_pre,
                                                                   rel_inds_graph=rel_inds,
                                                                   vr_graph=union_feat,
                                                                   rel_inds_offset_graph=rel_inds_offset,
                                                                   obj_preds=adj_obj_preds, num=0,
                                                                   )
                    if self.with_gt_adj_mat:
                        adj_dis_softmax_rel = gt_adj_mat
                    else:
                        adj_dis_softmax_rel = F.softmax(adj_dist_owbias_rel, -1)[:, 1]

            ''' '''
            if self.pass_in_obj_feats_to_gcn:
                pred_embed3 = self.obj_embed3(obj_preds_nozeros)
                pred_embed3 = F.dropout(pred_embed3, self.dropout_rate, training=self.training)
                obj_pre_rep3 = torch.cat((obj_fmaps, pred_embed3, pos_embed), 1)
                gtc_input = torch.cat((obj_pre_rep3, gtc_input), -1)
            if self.pass_embed_togcn:
                pred_embed3 = self.obj_embed3(obj_preds_nozeros)
                pred_embed3 = F.dropout(pred_embed3, self.dropout_rate, training=self.training)
                gtc_input = torch.cat((pred_embed3, gtc_input), -1)
            sub_gcn_ctx, obj_gcn_ctx = self.gcn_ctx(
                gtc_input,
                adj_mat=adj_dis_softmax_rel,  # if not self.training else gt_adj_mat
                im_inds=im_inds,
                rel_inds=rel_inds,
                rel_inds_offset=rel_inds_offset,
                num_layer=0,
                obj_preds=obj_preds_nozeros,
            )
            edge_sub_ctx = torch.cat([sub_gcn_ctx, edge_ctx], -1)
            edge_obj_ctx = torch.cat([obj_gcn_ctx, edge_ctx], -1)
            edge_ctx = sub_gcn_ctx + obj_gcn_ctx + edge_ctx

        return obj_dists2, obj_preds_nozeros, obj_preds_zeros, edge_ctx, edge_sub_ctx, edge_obj_ctx, \
               obj_feat_att, obj_feat_att_w, \
               adj_dist_owbias_rel, adj_dist_owbias_obj, keep_union, rel_inds_nms, \
               rel_dists


class RelModel(nn.Module):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=True, require_overlap_det=True,
                 embed_dim=200, hidden_dim=256, pooling_dim=2048,
                 nl_obj=1, nl_edge=2, nl_adj=2, nl_mul=0, use_resnet=False, order='confidence', thresh=0.01,
                 use_proposals=False, pass_in_obj_feats_to_decoder=True, pass_in_obj_feats_to_gcn=False,
                 pass_in_obj_feats_to_edge=True, pass_embed_togcn=False,
                 rec_dropout=0.0, use_bias=True, use_tanh=True,
                 limit_vision=True, attention_dim=256, adj_embed_dim=256, with_adj_mat=False,
                 max_obj_num=65, bg_num_graph=-1, bg_num_rel=-1, adj_embed=False, mean_union_feat=False,
                 ch_res=False, with_att=False, att_dim=512,
                 with_gcn=False, fb_thr=0.5, with_biliner_score=False,
                 gcn_adj_type='hard', where_gcn='parall', with_gt_adj_mat=False, type_gcn='normal',
                 edge_ctx_type='obj', nms_union=False, cosine_dis=False, test_alpha=0.5,
                 ext_feat=False,
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
        self.limit_vision = limit_vision
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'
        self.with_adj_mat = with_adj_mat
        self.bg_num_graph = bg_num_graph
        self.bg_num_rel = bg_num_rel
        self.max_obj_num = max_obj_num
        self.adj_embed_dim = adj_embed_dim
        self.nl_adj = nl_adj
        self.nl_mul = nl_mul
        self.ch_res = ch_res
        self.att_dim = att_dim
        self.fb_thr = fb_thr
        self.where_gcn = where_gcn
        self.dropout_rate = rec_dropout
        self.with_biliner_score = with_biliner_score
        self.with_gt_adj_mat = with_gt_adj_mat
        self.nl_obj = nl_obj
        self.with_gcn = with_gcn
        self.with_att = with_att
        self.type_gcn = type_gcn
        self.edge_ctx_type = edge_ctx_type
        self.nms_union = nms_union
        self.with_adaptive = False
        self.with_cosine_dis = cosine_dis
        self.test_alpha = test_alpha
        self.ext_feat = ext_feat
        if self.with_cosine_dis:
            print('With_cosine_dis ')
            self.obj_glove_vec = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
            self.rel_glove_vec = obj_edge_vectors(self.rel_classes, wv_dim=self.embed_dim)

        self.detector = ObjectDetector(
            classes=classes,
            mode=(
                'proposals' if use_proposals else 'refinerels') if mode == 'sgdet' or mode == 'predcls_nongtbox' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64,
            bg_num_graph=self.bg_num_graph,
            bg_num_rel=self.bg_num_rel,
            with_gt_adj_mat=with_gt_adj_mat,
        )
        if self.mode == 'detclass' and nl_obj == 0 and not with_gcn:
            return
        self.context = LinearizedContext(self.classes, self.rel_classes, mode=self.mode,
                                         embed_dim=self.embed_dim, hidden_dim=self.hidden_dim,
                                         pooling_dim=self.pooling_dim,
                                         obj_dim=self.obj_dim, nl_mul=nl_mul,
                                         nl_obj=nl_obj, nl_edge=nl_edge, nl_adj=nl_adj,
                                         dropout_rate=rec_dropout,
                                         order=order,
                                         pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                                         pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge,
                                         pass_in_obj_feats_to_gcn=pass_in_obj_feats_to_gcn,
                                         pass_embed_togcn=pass_embed_togcn,
                                         attention_dim=attention_dim, adj_embed_dim=self.adj_embed_dim,
                                         with_adj_mat=with_adj_mat, adj_embed=adj_embed,
                                         mean_union_feat=mean_union_feat, ch_res=ch_res,
                                         use_bias=use_bias,
                                         use_vision=use_vision,
                                         use_tanh=use_tanh,
                                         limit_vision=limit_vision,
                                         bg_num_rel=bg_num_rel,
                                         bg_num_graph=bg_num_graph,
                                         with_gcn=with_gcn,
                                         fb_thr=fb_thr,
                                         with_biliner_score=with_biliner_score,
                                         gcn_adj_type=gcn_adj_type,
                                         where_gcn=where_gcn,
                                         with_gt_adj_mat=with_gt_adj_mat,
                                         type_gcn=type_gcn,
                                         edge_ctx_type=edge_ctx_type,
                                         nms_union=nms_union,
                                         )

        # Image Feats (You'll have to disable if you want to turn off the features from here)
        # self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
        #                                      dim=1024 if use_resnet else 512)

        if use_resnet:

            roi_fmap = load_resnet(pretrained=False)[1]
            if pooling_dim != 2048:
                roi_fmap.append(nn.Linear(2048, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_resnet(pretrained=False)[1]
            self.compress_union = None
            self.compress = None
            '''
            roi_fmap = [Flattener(),
                        nn.Sequential(
                        nn.Linear(256 * 7 * 7, 2048),
                        nn.SELU(inplace=True),
                        nn.AlphaDropout(p=0.05),
                        nn.Linear(2048, 2048),
                        nn.SELU(inplace=True),
                        nn.AlphaDropout(p=0.05))]
            if pooling_dim != 2048:
                roi_fmap.append(nn.Linear(2048, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = nn.Sequential(
                        nn.Linear(256 * 7 * 7, 2048),
                        nn.SELU(inplace=True),
                        nn.AlphaDropout(p=0.05),
                        nn.Linear(2048, 2048),
                        nn.SELU(inplace=True),
                        nn.AlphaDropout(p=0.05))
            self.compress = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
            )
            self.compress_union = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(256),
            )
            '''
        else:

            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096,
                         pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
            self.compress_union = None
            self.compress = None

        self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                              dim=1024 if use_resnet else 512, use_resnet=self.use_resnet,
                                              compress_union=self.compress_union)

        ###################################
        post_lstm_in_dim = self.hidden_dim
        if self.with_gcn:
            post_lstm_in_dim += self.hidden_dim
            # post_lstm_in_dim = self.hidden_dim
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
            self.post_emb = nn.Embedding(self.num_classes, self.pooling_dim * 2)
            self.post_emb.weight.data.normal_(0, math.sqrt(1.0))

        if self.with_adaptive:
            self.adp_bilinear = nn.Linear(self.pooling_dim, 2, bias=False)
            self.adp_bilinear.weight = torch.nn.init.xavier_normal(self.adp_bilinear.weight, gain=1.0)

        if with_biliner_score:
            self.rel_bilinear = nn.Linear(self.pooling_dim, self.num_rels, bias=False)
            self.rel_bilinear.weight = torch.nn.init.xavier_normal(self.rel_bilinear.weight, gain=1.0)
        else:
            self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels, bias=True)
            self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)
        if self.use_bias and self.edge_ctx_type != 'union':
            self.freq_bias = FrequencyBias(with_bg=bg_num_rel != 0)

        if with_att:
            self.query_conv = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim // 8, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim // 8, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim, kernel_size=1)
            '''
            self.query_conv = nn.Linear(self.pooling_dim, self.pooling_dim // 8, bias=True)
            self.key_conv = nn.Linear(self.pooling_dim, self.pooling_dim // 8, bias=True)
            self.value_conv = nn.Linear(self.pooling_dim, self.pooling_dim, bias=True)
            '''
            self.gamma = nn.Parameter(torch.zeros(1))

            self.softmax = nn.Softmax(dim=-1)  #
            self.graph_att = None

            '''
            self.query_conv_1 = nn.Linear(self.pooling_dim, self.pooling_dim // 8, bias=True)
            self.key_conv_1 = nn.Linear(self.pooling_dim, self.pooling_dim // 8, bias=True)
            self.value_conv_1 = nn.Linear(self.pooling_dim, self.pooling_dim, bias=True)
            '''
            self.query_conv_1 = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim // 8,
                                          kernel_size=1)
            self.key_conv_1 = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim // 8, kernel_size=1)
            self.value_conv_1 = nn.Conv2d(in_channels=self.pooling_dim, out_channels=self.pooling_dim, kernel_size=1)

            self.gamma_1 = nn.Parameter(torch.zeros(1))

            self.softmax_1 = nn.Softmax(dim=-1)  #
            self.rel_att = None

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def self_attention_layer_graph(self, x):

        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        '''
        proj_query_t = self.query_conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_query = proj_query_t.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key_t = self.key_conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_key = proj_key_t.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value_t = self.value_conv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_value = proj_value_t.view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        '''
        return out.mean(3).mean(2), attention

    def self_attention_layer_rel(self, x):

        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv_1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv_1(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax_1(energy)  # BX (N) X (N)
        proj_value = self.value_conv_1(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma_1 * out + x
        '''
        m_batchsize, C, width, height = x.size()
        proj_query_t = self.query_conv_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_query = proj_query_t.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key_t = self.key_conv_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_key = proj_key_t.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax_1(energy)  # BX (N) X (N)
        proj_value_t = self.value_conv_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        proj_value = proj_value_t.view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma_1 * out + x
        '''
        return out.mean(3).mean(2), attention

    def visual_rep(self, features, rois, pair_inds, type='graph'):
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
        # return self.roi_fmap(uboxes)
        ''' '''
        if not self.use_resnet:
            return self.roi_fmap(uboxes)
        else:
            # print('uboxes: ',uboxes.size())
            roi_fmap_t = self.roi_fmap(uboxes)
            # print('roi_fmap_t: ',roi_fmap_t.size())
            if self.with_att:
                if type == 'graph':
                    output, self.graph_att = self.self_attention_layer_graph(roi_fmap_t)
                    return output
                if type == 'rel':
                    output, self.rel_att = self.self_attention_layer_rel(roi_fmap_t)
                    return output
            else:
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
            rel_inds_offset = rel_inds
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
            '''
            result.rel_labels_graph, result.gt_adj_mat_graph, result.rel_labels_offset_graph\
                = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1, max_obj_num=self.max_obj_num,
                                                time_bg=self.bg_num_graph)
            if self.bg_num_rel == self.bg_num_graph:
                result.rel_labels_rel, result.gt_adj_mat_rel, result.rel_labels_offset_rel = \
                result.rel_labels_graph.clone(), result.gt_adj_mat_graph.clone(), result.rel_labels_offset_graph.clone()
            else:
                result.rel_labels_rel, result.gt_adj_mat_rel, result.rel_labels_offset_rel\
                    = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1, max_obj_num=self.max_obj_num,
                                                time_bg=self.bg_num_rel)
            '''
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
        else:
            result.gt_adj_mat_graph = gt_adj_mat
        # print('result.rel_labels_graph',result.rel_labels_graph)
        # print('result.rel_labels_offset_graph', result.rel_labels_offset_graph)
        # print('result.rel_labels_rel', result.rel_labels_rel)
        # print('result.rel_labels_offset_rel', result.rel_labels_offset_rel)
        # print('result.gt_adj_mat_graph', result.gt_adj_mat_graph)
        if self.mode != 'predcls_nongtbox':
            rel_inds_graph, rel_inds_offset_graph = self.get_rel_inds(result.rel_labels_graph,
                                                                      result.rel_labels_offset_graph, im_inds, boxes)

            rel_inds_rel, rel_inds_offset_rel = self.get_rel_inds(result.rel_labels_rel, result.rel_labels_offset_rel,
                                                                  im_inds, boxes)
        # rel_inds: shape [num_rels, 3], each array is [img_ind, box_ind1, box_ind2] for training
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # Prevent gradients from flowing back into score_fc from elsewhere
        if (self.training or self.mode == 'predcls') and self.mode != 'predcls_nongtbox':
            obj_labels = result.rm_obj_labels
        elif self.mode == 'predcls_nongtbox':
            rois, obj_labels, _, _, _, rel_labels = self.detector.gt_boxes(None, im_sizes, image_offset, gt_boxes,
                                                                           gt_classes, gt_rels, train_anchor_inds,
                                                                           proposals=proposals)
            im_inds = rois[:, 0].long().contiguous()
            gt_boxes = rois[:, 1:]
            rel_inds = self.get_rel_inds(rel_labels, im_inds, gt_boxes)
            result.rel_labels = rel_labels
            result.rm_obj_labels = obj_labels

        else:
            obj_labels = None

        vr_graph = self.visual_rep(result.fmap.detach(), rois, rel_inds_graph[:, 1:], type='graph')
        vr_rel = self.visual_rep(result.fmap.detach(), rois, rel_inds_rel[:, 1:], type='rel')

        if result.rel_labels_graph is None or (not self.training):
            rel_label = None
        else:
            rel_label = result.rel_labels_graph[:, -1]
        '''         '''
        if result.gt_adj_mat_graph is not None:
            gt_adj_mat = result.gt_adj_mat_graph[rel_inds_graph[:, 1], \
                                                 rel_inds_offset_graph[:, 2]].type(torch.cuda.LongTensor).type_as(
                result.fmap.data)
        else:
            gt_adj_mat = None

        result.rm_obj_dists, result.obj_preds_nozeros, result.obj_preds_zeros, edge_ctx, edge_sub_ctx, edge_obj_ctx, obj_feat_att, obj_feat_att_w, \
        adj_mat_rel, adj_mat_obj, keep_union, rel_inds_nms, rel_dists = self.context(
            result.obj_fmap,
            result.rm_obj_dists.detach(),
            im_inds, obj_labels,
            boxes.data, result.boxes_all, obj_feat_im_inds, result.fmap,
            union_feat=vr_graph, rel_inds=rel_inds_graph.clone(), rel_inds_offset=rel_inds_offset_graph,
            gt_adj_mat=gt_adj_mat,
            rel_label=rel_label,
            label_prior=result.obj_preds,
        )
        result.att_alpha = obj_feat_att_w
        result.im_inds = im_inds
        result.obj_feat_im_inds = obj_feat_im_inds
        result.pre_adj_mat_rel = adj_mat_rel
        result.pre_adj_mat_obj = adj_mat_obj
        result.keep_union = keep_union
        if self.mode == 'detclass':
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
        if self.edge_ctx_type != 'union':

            if edge_ctx is None:
                edge_rep = self.post_emb(result.obj_preds)
            else:
                subj_rep = self.post_sub_lstm(edge_sub_ctx)
                obj_rep = self.post_obj_lstm(edge_obj_ctx)
            # Split into subject and object representations
            # edge_rep = edge_rep.view(edge_rep.size(0), 2, self.pooling_dim)
            subj_rep = F.dropout(subj_rep, self.dropout_rate, training=self.training)
            obj_rep = F.dropout(obj_rep, self.dropout_rate, training=self.training)
            # subj_rep = edge_rep[:, 0]
            # obj_rep = edge_rep[:, 1]
            obj_rel_ind = rel_inds_rel
            vr_obj = vr_rel
            subj_rep_rel = subj_rep[obj_rel_ind[:, 1]]
            obj_rep_rel = obj_rep[obj_rel_ind[:, 2]]
            # prod_rep = subj_rep[obj_rel_ind[:, 1]] * obj_rep[obj_rel_ind[:, 2]]
            if self.use_vision:
                if self.mode != 'predcls_nongtbox':
                    vr_obj = vr_obj
                else:
                    vr_obj = 0.5 * (obj_feat_att[obj_rel_ind[:, 1]] + obj_feat_att[obj_rel_ind[:, 2]])
                if self.limit_vision:
                    # exact value TBD
                    subj_rep_rel = torch.cat((subj_rep_rel[:, :2048] * subj_rep_rel[:, :2048], subj_rep_rel[:, 2048:]),
                                             1)
                    obj_rep_rel = torch.cat((obj_rep_rel[:, :2048] * obj_rep_rel[:, :2048], obj_rep_rel[:, 2048:]), 1)
                else:
                    subj_rep_rel = subj_rep_rel * vr_obj[:, :self.pooling_dim]
                    obj_rep_rel = obj_rep_rel * vr_obj[:, :self.pooling_dim]

            if self.use_tanh:
                subj_rep_rel = F.tanh(subj_rep_rel)
                obj_rep_rel = F.tanh(obj_rep_rel)
            if self.with_biliner_score:
                result.rel_dists = self.bilinear_score(subj_rep_rel, obj_rep_rel)
            else:
                prod_rep = subj_rep_rel * obj_rep_rel
                result.rel_dists = self.rel_compress(prod_rep)
        elif self.edge_ctx_type == 'union':
            obj_rel_ind = rel_inds_nms
            result.rel_dists = rel_dists
            rel_inds_rel = rel_inds_nms
            result.rel_labels_rel = result.rel_labels_graph
            if keep_union is not None:
                result.rel_labels_rel = result.rel_labels_graph[keep_union]
        if self.use_bias and self.edge_ctx_type != 'union':
            if self.mode != 'sgdet':
                rel_obj_preds = result.obj_preds_nozeros.clone()
            else:

                rel_obj_preds = result.obj_preds_nozeros.clone()
                '''
                if self.training:
                    rel_obj_preds = result.rm_obj_labels.clone()
                else:
                    rel_obj_preds = result.rm_obj_dists.max(1)[1]
                '''

            if self.with_adaptive:
                sen_vis_score = self.adp_bilinear_score(subj_rep_rel * obj_rep_rel,
                                                        vr_obj[:, :self.pooling_dim])

                result.rel_dists = sen_vis_score[:, 0, None] * result.rel_dists \
                                   + sen_vis_score[:, 1, None] * \
                                   self.freq_bias.index_with_labels(torch.stack((
                                       rel_obj_preds[obj_rel_ind[:, 1]],
                                       rel_obj_preds[obj_rel_ind[:, 2]],
                                   ), 1))
            else:
                result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                    rel_obj_preds[obj_rel_ind[:, 1]],
                    rel_obj_preds[obj_rel_ind[:, 2]],
                ), 1))

        if self.training:
            if self.with_cosine_dis:
                self.obj_glove_vec = self.obj_glove_vec.contiguous().cuda(obj_rel_ind.get_device(), async=True)
                self.rel_glove_vec = self.rel_glove_vec.contiguous().cuda(obj_rel_ind.get_device(), async=True)
                fg_ind = result.num_true_rel.data

                obj_label_1 = result.rm_obj_labels[obj_rel_ind[fg_ind][:, 1]].data
                obj_label_2 = result.rm_obj_labels[obj_rel_ind[fg_ind][:, 2]].data
                rel_label = result.rel_labels_rel[fg_ind][:, -1].data

                sub_glove_vec = self.obj_glove_vec[obj_label_1]
                obj_glove_vec = self.obj_glove_vec[obj_label_2]
                rel_glove_vec = self.rel_glove_vec[rel_label]
                all_glove_vec = torch.cat([sub_glove_vec, rel_glove_vec, obj_glove_vec], -1)
                all_rel_num = all_glove_vec.shape[0]
                all_rel_ind = np.arange(all_rel_num)
                all_rel_mat = np.ones([all_rel_num, all_rel_num])
                all_rel_mat[all_rel_ind, all_rel_ind] = 0.
                all_rel_ind_1, all_rel_ind_2 = all_rel_mat.nonzero()
                all_rel_ind_1 = torch.LongTensor(all_rel_ind_1).contiguous().cuda(obj_rel_ind.get_device(), async=True)
                all_rel_ind_2 = torch.LongTensor(all_rel_ind_2).contiguous().cuda(obj_rel_ind.get_device(), async=True)
                all_glove_cosine_dis = F.cosine_similarity(all_glove_vec[all_rel_ind_1], all_glove_vec[all_rel_ind_2])
                all_rep = (subj_rep_rel * obj_rep_rel)[fg_ind]
                all_rep_cosine_dis = F.cosine_similarity(all_rep[all_rel_ind_1], all_rep[all_rel_ind_2])
                result.all_rep_glove_rate = all_rep_cosine_dis / (Variable(all_glove_cosine_dis) + 1e-8)

            return result

        twod_inds = arange(result.obj_preds_nozeros.data) * self.num_classes + result.obj_preds_nozeros.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]
        if result.pre_adj_mat_rel is not None:
            result.pre_adj_mat_rel = F.softmax(result.pre_adj_mat_rel, dim=-1)
        if result.pre_adj_mat_obj is not None:
            result.pre_adj_mat_obj = F.softmax(result.pre_adj_mat_obj, dim=-1)

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        elif self.mode == 'predcls_nongtbox':
            bboxes = gt_boxes
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)

        return filter_dets(ids, im_inds, bboxes, result.obj_scores,
                           result.obj_preds_nozeros,
                           result.pre_adj_mat_rel[:, 1] if result.pre_adj_mat_rel is not None else None,
                           result.pre_adj_mat_obj[:, 1] if result.pre_adj_mat_obj is not None else None,
                           rel_inds_rel[:, 1:],
                           rel_rep,
                           nongt_box=self.mode == 'predcls_nongtbox', with_adj_mat=self.with_adj_mat,
                           with_gt_adj_mat=self.with_gt_adj_mat, gt_adj_mat=gt_adj_mat,
                           alpha=self.test_alpha, feat=subj_rep_rel * obj_rep_rel,
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
