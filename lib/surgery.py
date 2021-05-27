# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time

def ass_embed_dist(ass_embed):
    num = ass_embed.size()[0]
    size = (num, num, ass_embed.size()[1])
    ass_embed_a = ass_embed.unsqueeze(dim=1).expand(*size)
    ass_embed_b = ass_embed_a.permute(1, 0, 2)
    diff_push = (ass_embed_a - ass_embed_b)
    diff_push = torch.pow(diff_push, 2).sum(dim=2)
    diff_push_t = (diff_push - diff_push.min()) / (diff_push.max() - diff_push.min())
    return  diff_push_t.max() - diff_push_t

def filter_dets(ids, im_inds, boxes, obj_scores, obj_classes, adj_mat_rel=None, adj_mat_obj=None,
                rel_inds=None, pred_scores=None,
                nongt_box = False, with_adj_mat=False, with_gt_adj_mat=False, gt_adj_mat=None,
                alpha=0.5, feat=None, ext_feat=False, mode=None, beta=0.5,
                ass_embed_obj=None, ass_embed_rel=None):
    """
    Filters detections....
    when testing model, the batch size always equal to one.
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param adj_mat: [num_box, 65] adj matrix
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0)
    if nongt_box == False:
        assert obj_scores.size(0) == num_box

    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:,0]]
    obj_scores1 = obj_scores.data[rel_inds[:,1]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:,1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

    if with_adj_mat:
        '''
        rel_adj_mat = torch.zeros_like(adj_mat)
        rel_adj_mat[rel_inds[:,0],rel_inds[:,1]] = 1.0
        adj_mat = adj_mat * rel_adj_mat
        adj_mat = adj_mat[:,: adj_mat.size(0)].data
        im_inds = im_inds.data
        bool_adj_mat = im_inds[:,None] == im_inds[None,:]
        bool_adj_mat = bool_adj_mat
        dual_ind = torch.arange(adj_mat.size(0))
        dual_ind = dual_ind[:,None] == dual_ind[None,:]
        dual_ind = dual_ind.type_as(adj_mat)
        dual_ind = 1.0 - dual_ind
        bool_adj_mat = bool_adj_mat.type_as(adj_mat)
        adj_mat = adj_mat * bool_adj_mat
        adj_mat = adj_mat * dual_ind
        '''
        if adj_mat_rel is not None:
            adj_mat_rel = adj_mat_rel.data.contiguous()
        else:
            adj_mat_rel = None

        if adj_mat_obj is not None:
            adj_mat_obj = adj_mat_obj.data.contiguous()
        else:
            adj_mat_obj = None
        #adj_mat = adj_mat[rel_inds[:,0],rel_inds[:,1]]

        if with_gt_adj_mat:
            rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1 * gt_adj_mat.data.contiguous()
            adj_mat_obj = gt_adj_mat.data.contiguous()
            adj_mat_rel = gt_adj_mat.data.contiguous()
        else:
            if mode != 'sgdet':
                if adj_mat_obj is None and adj_mat_rel is not None:
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                          ((adj_mat_rel) ** (alpha))
                elif adj_mat_rel is None and adj_mat_obj is not None:
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                          ((adj_mat_obj) ** (alpha))
                elif adj_mat_rel is not None and adj_mat_obj is not None:
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                          (((adj_mat_rel) * (adj_mat_obj)) ** (alpha))
                else:
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) )

            else:
                if adj_mat_obj is not None:
                    '''
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                          ((adj_mat_rel * adj_mat_obj) ** (alpha))
                    '''
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                      (((adj_mat_rel**(beta)) * (adj_mat_obj**((1.0 - beta)))) ** (alpha))

                else:
                    rel_scores_argmaxed = ((pred_scores_max * obj_scores0 * obj_scores1) ** (1.0 - alpha)) * \
                                          ((adj_mat_rel) ** (alpha))
            '''
            if ass_embed_obj is not None and ass_embed_rel is not None:
                ass_embed_obj_dist = (ass_embed_dist(ass_embed_obj)[rel_inds[:,0],rel_inds[:,1]]).data.contiguous()
                ass_embed_rel_dist = (ass_embed_dist(ass_embed_rel)[rel_inds[:,0],rel_inds[:,1]]).data.contiguous()
                rel_scores_argmaxed = rel_scores_argmaxed * ass_embed_obj_dist * ass_embed_rel_dist
            '''
        rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

    rels = rel_inds[rel_scores_idx].cpu().numpy()

    pred_scores_sorted = pred_scores[rel_scores_idx].data.cpu().numpy()
    obj_scores_np = obj_scores.data.cpu().numpy()
    objs_np = obj_classes.data.cpu().numpy()
    boxes_out = boxes.data.cpu().numpy()

    if adj_mat_rel is not None:
        adj_mat_rel_np = adj_mat_rel[rel_scores_idx].cpu().numpy()
    else:
        adj_mat_rel_np = None

    if adj_mat_obj is not None:
        adj_mat_obj_np = adj_mat_obj[rel_scores_idx].cpu().numpy()
    else:
        adj_mat_obj_np = None

    if gt_adj_mat is not None:
        gt_adj_mat_np = gt_adj_mat[rel_scores_idx].data.cpu().numpy()
    else:
        gt_adj_mat_np = None
    if ext_feat:
        feat_np = feat[rel_scores_idx].data.cpu().numpy()
        return ids, boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted, \
               adj_mat_rel_np, adj_mat_obj_np, gt_adj_mat_np, feat_np
    return boxes_out, objs_np, obj_scores_np, rels, pred_scores_sorted, \
           adj_mat_rel_np, adj_mat_obj_np, gt_adj_mat_np

# def _get_similar_boxes(boxes, obj_classes_topk, nms_thresh=0.3):
#     """
#     Assuming bg is NOT A LABEL.
#     :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
#     :param obj_classes: [num_box, topk] class labels
#     :return: num_box, topk, num_box, topk array containing similarities.
#     """
#     topk = obj_classes_topk.size(1)
#     num_box = boxes.size(0)
#
#     box_flat = boxes.view(-1, 4) if boxes.dim() == 3 else boxes[:, None].expand(
#         num_box, topk, 4).contiguous().view(-1, 4)
#     jax = bbox_overlaps(box_flat, box_flat).data > nms_thresh
#     # Filter out things that are not gonna compete.
#     classes_eq = obj_classes_topk.data.view(-1)[:, None] == obj_classes_topk.data.view(-1)[None, :]
#     jax &= classes_eq
#     boxes_are_similar = jax.view(num_box, topk, num_box, topk)
#     return boxes_are_similar.cpu().numpy().astype(np.bool)
