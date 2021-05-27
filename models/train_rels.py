"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""
import matplotlib
matplotlib.use('Agg')
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.evaluation.sg_eval_all_rel_cates import BasicSceneGraphEvaluator as BasicSceneGraphEvaluator_rel
from lib.pytorch_misc import print_para
from lib.pytorch_misc import load_reslayer4
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.loss.AE_loss import AE_loss
from lib.loss.Focal_loss import FocalLoss
import random
conf = ModelConfig()
if conf.model == 'motifnet':
    #from lib.rel_model_AE2lstm import RelModel
    #from lib.rel_model_linknet import RelModel
    #from lib.rel_model_neg import RelModel
    #from lib.ablation_study.rel_model_noadjloss import RelModel
    #from lib.ablation_study.rel_model_edgelstm_regfeat import RelModel
    #from lib.ablation_study.rel_model_gcn_front import RelModel
    #from lib.ablation_study.rel_model_nogcn import RelModel
    #from lib.ablation_study.rel_model_nolstm2 import RelModel
    #from lib.rel_model_topgcn import RelModel
    from lib.models.rel_model_bert import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet',
                          keep_pred=conf.keep_pred,
                          )
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
                                               
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj,
                    nh_obj=conf.nh_edge, nh_edge=conf.nh_edge,
                    nl_adj=conf.nl_adj,
                    hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pass_in_obj_feats_to_gcn=conf.pass_in_obj_feats_to_gcn,
                    pass_embed_togcn=conf.pass_embed_togcn,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    limit_vision=conf.limit_vision,
                    attention_dim=conf.attention_dim,
                    adj_embed_dim = conf.adj_embed_dim,
                    with_adj_mat=conf.with_adj_mat,
                    bg_num_graph=conf.bg_num_graph,
                    bg_num_rel=conf.bg_num_rel,
                    neg_time=conf.neg_time,
                    adj_embed=conf.adj_embed,
                    mean_union_feat=conf.mean_union_feat,
                    ch_res=conf.ch_res,
                    with_att=conf.with_att,
                    with_gcn=conf.with_gcn,
                    fb_thr=conf.fb_thr,
                    with_biliner_score=conf.with_biliner_score,
                    gcn_adj_type=conf.gcn_adj_type,
                    where_gcn=conf.where_gcn,
                    with_gt_adj_mat=conf.gt_adj_mat,
                    type_gcn=conf.type_gcn,
                    edge_ctx_type=conf.edge_ctx_type,
                    nms_union=conf.nms_union,
                    cosine_dis=conf.cosine_dis,
                    test_alpha=conf.test_alpha,
                    )

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    # params = [p for n,p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.l2, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1,)
    return optimizer, scheduler



ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])
    #print('detector: ',detector.detector)
    # for i in ckpt['state_dict'].keys():
        # if 'roi_fmap' in i:
            # print('ckpt state_dict: ',i) 
    if not conf.use_resnet:
        detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
        detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
        detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
        detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
        detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
    else :
        load_reslayer4(detector, ckpt, 3)
    """
    if conf.use_resnet:
        detector.compress[0].weight.data.copy_(ckpt['state_dict']['compress.0.weight'])
        detector.compress[0].bias.data.copy_(ckpt['state_dict']['compress.0.bias'])
        detector.compress[2].weight.data.copy_(ckpt['state_dict']['compress.2.weight'])
        detector.compress[2].bias.data.copy_(ckpt['state_dict']['compress.2.bias'])
        detector.union_boxes.compress_union[0].weight.data.copy_(ckpt['state_dict']['compress.0.weight'])
        detector.union_boxes.compress_union[0].bias.data.copy_(ckpt['state_dict']['compress.0.bias'])
        detector.union_boxes.compress_union[2].weight.data.copy_(ckpt['state_dict']['compress.2.weight'])
        detector.union_boxes.compress_union[2].bias.data.copy_(ckpt['state_dict']['compress.2.bias'])
    """
detector.cuda()
#Rel_loss = FocalLoss(gamma=2,alpha=None)
#Rel_loss.cuda()
#Class_loss = FocalLoss(gamma=2,alpha=None)
#Class_loss.cuda()
#Graph_loss = FocalLoss(gamma=2,alpha=0.25)
#Graph_loss.cuda()

def train_epoch(epoch_num):
    detector.train()
    tr = []
    for param_group in optimizer.param_groups:
        print('learning rate: ', param_group['lr'])
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)

            start = time.time()
            if conf.debug:
                mAp = val_epoch()
                detector.train()
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return pd.concat(tr, axis=1)


def balance_graph_loss(pre, gt):
    posi_rate = gt.sum() / (gt.shape[0] * 1.0)
    negt_rate = 1.0 - posi_rate
    posi_w = torch.FloatTensor(np.array([0.0, 1.0])).cuda(pre.get_device())
    negt_w = torch.FloatTensor(np.array([1.0, 0.0])).cuda(pre.get_device())
    posi_loss = negt_rate * F.cross_entropy(pre, gt, weight=posi_w)
    negt_loss = posi_rate * F.cross_entropy(pre, gt, weight=negt_w)
    return  posi_loss + negt_loss

def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    if conf.mode == 'detclass' :
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        if conf.with_adj_mat and not conf.gt_adj_mat:
            # result.pre_adj_mat = result.pre_adj_mat.view([result.pre_adj_mat.size(0)*result.pre_adj_mat.size(1),-1])
            # result.gt_adj_mat = result.gt_adj_mat.view([-1])
            loss_in_pre_bg_score = result.pre_adj_mat
            loss_in_gt_bg_score = result.gt_adj_mat_graph[result.rel_labels_graph[:, 1], \
                                                          result.rel_labels_offset_graph[:, 2]].type(torch.cuda.LongTensor)

            losses['graph_loss'] = F.cross_entropy(loss_in_pre_bg_score, loss_in_gt_bg_score)
        loss = sum(losses.values())
    else :

        if conf.with_adj_mat and not conf.gt_adj_mat:

            if result.pre_adj_mat_rel is not None:

                loss_in_pre_bg_score_rel = result.pre_adj_mat_rel
                loss_in_gt_bg_score = result.gt_adj_mat_graph[result.rel_labels_graph[:, 1], \
                                                        result.rel_labels_offset_graph[:, 2]].type(torch.cuda.LongTensor)
                loss_in_gt_bg_1 = ((loss_in_gt_bg_score.type_as(loss_in_pre_bg_score_rel)).sum()) / (1.0*loss_in_gt_bg_score.size(0))
                loss_in_gt_bg_0 = 1.0 - loss_in_gt_bg_1
                loss_w = torch.cat([loss_in_gt_bg_1[None], loss_in_gt_bg_0[None]])

                losses['graph_loss_rel'] = F.cross_entropy(loss_in_pre_bg_score_rel, loss_in_gt_bg_score)
                #losses['graph_loss_rel'] = Graph_loss(loss_in_pre_bg_score_rel,loss_in_gt_bg_score)
            if result.pre_adj_mat_obj is not None:
                loss_in_pre_bg_score_obj = result.pre_adj_mat_obj
                loss_in_gt_bg_score = result.gt_adj_mat_graph[result.rel_labels_graph[:, 1], \
                                                              result.rel_labels_offset_graph[:, 2]].type(torch.cuda.LongTensor)
                #print('loss_in_gt_bg_score:',loss_in_gt_bg_score)
                #print('loss_in_pre_bg_score_1: ',loss_in_pre_bg_score_1)
                losses['graph_loss_obj'] = F.cross_entropy(loss_in_pre_bg_score_obj, loss_in_gt_bg_score)
                #losses['graph_loss_obj'] = Graph_loss(loss_in_pre_bg_score_obj, loss_in_gt_bg_score)
        if conf.cosine_dis:
            losses['cosine_dis'] = 10.0 * torch.var(result.all_rep_glove_rate, unbiased=False)

        #losses['ae_loss_obj'] = AE_loss(result.ass_embed_obj, result.gt_adj_mat_graph, result)
        #losses['ae_loss_rel'] = AE_loss(result.ass_embed_rel, result.gt_adj_mat_graph, result)
        #print('result.rm_obj_dists: ',result.rm_obj_dists)
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
        #losses['class_loss'] = Class_loss(result.rm_obj_dists, result.rm_obj_labels)
        #losses['mul_class_loss'] = F.binary_cross_entropy(result.mul_dist, result.gt_mul_label)
        #print('class_loss: ', losses['class_loss'])
        if conf.neg_time > 0:
            losses['rel_loss'] = F.cross_entropy(result.rel_dists[result.id_tp_rel.data],
                                                 result.rel_labels_rel[result.id_tp_rel.data][:,-1])
            if result.id_neg_rel.size(0)>1:
                losses['rel_loss'] = losses['rel_loss']\
                                     + F.cross_entropy(-result.rel_dists[result.id_neg_rel[1:].data],
                                                     result.rel_labels_rel[result.id_neg_rel[1:].data][:,-1])
        else:
            losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels_rel[:, -1])
            #losses['rel_loss'] = Rel_loss(result.rel_dists, result.rel_labels_rel[:, -1])
        #losses['mul_rel_loss'] = F.binary_cross_entropy(result.mul_rel_dist, result.gt_mul_rel)
        #print('rel_loss: ', losses['rel_loss'])

        loss = sum(losses.values())

    optimizer.zero_grad()

    loss.backward()

    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch(epoch='final'):
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    evaluator_rel = BasicSceneGraphEvaluator_rel.all_modes()
    num_correct = 0
    num_sample = 0
    num_correct_adj_rel = 0
    num_correct_adj_obj = 0
    num_sample_adj = 0
    True_sample = 0
    TP_sample_obj = 0
    TP_sample_rel = 0
    cro_en = np.array([])
    for val_b, batch in enumerate(val_loader):
        num_correct_i, num_sample_i, cro_en_i,  num_correct_adj_obj_i, num_correct_adj_rel_i, \
        num_sample_adj_i, True_sample_i, TP_sample_obj_i, TP_sample_rel_i \
            = val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_rel)
        num_correct = num_correct + num_correct_i
        num_sample = num_sample + num_sample_i
        num_correct_adj_rel = num_correct_adj_rel + num_correct_adj_rel_i
        num_correct_adj_obj = num_correct_adj_obj + num_correct_adj_obj_i
        num_sample_adj = num_sample_adj + num_sample_adj_i
        cro_en = np.concatenate([cro_en, cro_en_i])
        True_sample = True_sample + True_sample_i
        TP_sample_obj = TP_sample_obj + TP_sample_obj_i
        TP_sample_rel = TP_sample_rel + TP_sample_rel_i
    evaluator[conf.mode].print_stats()
    evaluator_rel[conf.mode].print_stats(epoch)
    print('obj acc: ',(1.0 * num_correct) / (1.0 * num_sample))
    print('adj rel acc: ', (1.0 * num_correct_adj_rel) / (1.0 * num_sample_adj))
    print('adj obj acc: ', (1.0 * num_correct_adj_obj) / (1.0 * num_sample_adj))
    print('TP adj rel recall:', (TP_sample_rel * 1.0) / (True_sample * 1.0))
    print('TP adj obj recall:', (TP_sample_obj * 1.0) / (True_sample * 1.0))
    print('loss: ', -cro_en.mean())
    torch.cuda.empty_cache()
    if conf.mode == 'detclass':
        return None, \
               None, \
               None, \
               (1.0 * num_correct) / (1.0 * num_sample), \
               -cro_en.mean(), \
               (1.0 * num_correct_adj_obj) / (1.0 * num_sample_adj),\
               (1.0 * num_correct_adj_rel) / (1.0 * num_sample_adj), \
               (TP_sample_rel * 1.0) / (True_sample * 1.0), \
               (TP_sample_obj * 1.0) / (True_sample * 1.0)

    else:
        return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][20]), \
               np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][50]), \
               np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100]), \
               (1.0 * num_correct) / (1.0 * num_sample), \
               -cro_en.mean(), \
               (1.0 * num_correct_adj_obj) / (1.0 * num_sample_adj),\
               (1.0 * num_correct_adj_rel) / (1.0 * num_sample_adj), \
               (TP_sample_rel * 1.0) / (True_sample * 1.0), \
               (TP_sample_obj * 1.0) / (True_sample * 1.0)


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets*np.log(predictions+1e-9)))/N
    return ce

def val_batch(batch_num, b, evaluator, evaluator_rel):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]
    num_correct = 0
    num_sample = 0
    num_correct_adj_mat_rel = 0
    num_correct_adj_mat_obj = 0
    num_sample_adj_mat = 0
    TP_sample_obj = 0
    TP_sample_rel = 0
    True_sample = 0
    cro_en = np.array([])
    if conf.mode != 'detclass':
        for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, pred_adj_mat_rel_i,
                pred_adj_mat_obj_i, gt_adj_mat_i) in enumerate(det_res):
            gt_entry = {
                'gt_classes': val.gt_classes[batch_num + i].copy(),
                'gt_relations': val.relationships[batch_num + i].copy(),
                'gt_boxes': val.gt_boxes[batch_num + i].copy(),
            }
            assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
            num_sample = num_sample + objs_i.shape[0]
            num_sample_adj_mat = num_sample_adj_mat + gt_adj_mat_i.shape[0]
            True_sample = True_sample + (gt_adj_mat_i == 1).sum()
            if conf.mode !=  'sgdet':
                num_correct = num_correct + np.sum((val.gt_classes[batch_num + i].copy() - objs_i) == 0)
                if pred_adj_mat_rel_i is not None:
                    pred_adj_mat_bool_rel_i = (pred_adj_mat_rel_i > 0.5).astype('float64')
                else:
                    pred_adj_mat_bool_rel_i = 0
                if pred_adj_mat_obj_i is not None:
                    pred_adj_mat_bool_obj_i = (pred_adj_mat_obj_i > 0.5).astype('float64')
                else:
                    pred_adj_mat_bool_obj_i = 0

                num_correct_adj_mat_rel = num_correct_adj_mat_rel \
                                          + np.sum((gt_adj_mat_i - pred_adj_mat_bool_rel_i)==0)
                num_correct_adj_mat_obj = num_correct_adj_mat_obj \
                                          + np.sum((gt_adj_mat_i - pred_adj_mat_bool_obj_i)==0)
                TP_sample_obj = TP_sample_obj + np.sum(
                    ((gt_adj_mat_i - pred_adj_mat_bool_obj_i) == 0) * gt_adj_mat_i)
                TP_sample_rel = TP_sample_rel + np.sum(
                    ((gt_adj_mat_i - pred_adj_mat_bool_rel_i) == 0) * gt_adj_mat_i)
            else :
                correct_num = 0.0

            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,  # hack for now.
                'pred_adj_mat_rel': pred_adj_mat_rel_i,
                'pred_adj_mat_obj': pred_adj_mat_obj_i,
            }

            evaluator[conf.mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )
            evaluator_rel[conf.mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )
    else :
        for i, (boxes_i, objs_i, obj_scores_i) in enumerate(det_res):
            num_sample = num_sample + objs_i.shape[0]
            j = np.arange(obj_scores_i.shape[0])
            cro_en_i = np.log(obj_scores_i[j, val.gt_classes[batch_num + i]] + 1e-9)
            num_correct = num_correct + np.sum((val.gt_classes[batch_num + i].copy() - objs_i) == 0)
            cro_en = np.concatenate([cro_en,cro_en_i], -1)
    return num_correct, num_sample, cro_en,  num_correct_adj_mat_obj, num_correct_adj_mat_rel, \
           num_sample_adj_mat, True_sample, TP_sample_obj, TP_sample_rel

print("Training starts now!")
optimizer, scheduler = get_optim(conf.lr)
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    R_20, R_50, R_100, obj_acc, cro_en, adj_obj_acc, adj_rel_acc, \
    adj_obj_recall, adj_rel_recall = val_epoch(epoch)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            'R_20': R_20,
            'R_50': R_50,
            'R_100': R_100,
            'obj_acc': obj_acc,
            'cro_en':cro_en,
            'adj_obj_acc':adj_obj_acc,
            'adj_rel_acc': adj_rel_acc,
            'adj_obj_recall':adj_obj_recall,
            'adj_rel_recall':adj_rel_recall
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))
    if conf.mode != 'detclass':
        scheduler.step(R_100)
    else:
        scheduler.step(obj_acc)

    #if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
    #    print("exiting training early", flush=True)
    #    break
