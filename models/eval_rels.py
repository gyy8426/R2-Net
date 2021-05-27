import matplotlib
matplotlib.use('Agg')
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from functools import reduce
import torch
from sklearn.metrics import accuracy_score
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.evaluation.sg_eval_all_rel_cates import BasicSceneGraphEvaluator as BasicSceneGraphEvaluator_rel
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE, DATA_PATH
import dill as pkl
import os
from collections import defaultdict
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from lib.pytorch_misc import load_reslayer4
size_index = np.load('/home/guoyuyu/guoyuyu/code/code_by_myself/scene_graph/dataset_analysis/size_index.npy')
AE_results = pkl.load(open('/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn/AE_loss_sgcls','rb'))
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.models.rel_model_bert import RelModel
    #from lib.rel_model_linknet import RelModel
    #from lib.rel_model_complex_emb import RelModel
    #from lib.rel_model_adj1 import RelModel
    #from lib.rel_model_topgcn import RelModel
    #from lib.ablation_study.rel_model_notop2 import RelModel
    #from lib.ablation_study.rel_model_edgelstm_regfeat import RelModel
    #from lib.rel_model_AE import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet',
                          keep_pred=conf.keep_pred)
if conf.test:
    val = test
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


detector.cuda()
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
    if conf.mode != 'detclass':
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

#optimistic_restore(detector, ckpt['state_dict'])
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

all_pred_entries = []
all_TP_label_num = np.zeros([detector.num_classes])
all_label_num = np.zeros([detector.num_classes])
all_TP_size_num = np.zeros([size_index.shape[0]])
all_size_num = np.zeros([size_index.shape[0]])
all_pred_size = []

all_TP_rel_rel_num = np.zeros([detector.num_rels])
all_TP_rel_obj_num = np.zeros([detector.num_rels])
all_rel_num =  np.zeros([detector.num_rels])


all_TP_pred_num = np.zeros([3, detector.num_rels])
all_pred_num = np.zeros([detector.num_rels])
all_pred_recall = []
def count_num(label_i):
    for i in range(label_i.shape[0]):
        all_label_num[label_i[i]] = all_label_num[label_i[i]] + 1

def TP_count_num(label_i, pred_i):
    TP_labe_ind = ((label_i - pred_i) == 0)
    TP_labe = TP_labe_ind * pred_i
    for i in range(TP_labe.shape[0]):
        if TP_labe_ind[i]:
            all_TP_label_num[label_i[i]] = all_TP_label_num[label_i[i]] + 1

def count_size_num(boxes_i, image_size):
    size_i = abs(boxes_i[:, 2] - boxes_i[:, 0]) * abs(boxes_i[:, 3] - boxes_i[:, 1])
    size_i = size_i / (1.0 * image_size[0] * image_size[1])
    for i in range(size_i.shape[0]):
        ind = int((size_i[i] - size_index[0])/ (size_index[1]-size_index[0]))
        all_size_num[ind] = all_size_num[ind] + 1

def TP_count_size_num(label_i, pred_i, boxes_i, image_size):
    TP_labe_ind = ((label_i - pred_i) == 0)
    size_i = abs(boxes_i[:, 2] - boxes_i[:, 0]) * abs(boxes_i[:, 3] - boxes_i[:, 1])
    size_i = size_i / (1.0 * image_size[0] * image_size[1])
    for i in range(TP_labe_ind.shape[0]):
        if TP_labe_ind[i]:
            ind = int((size_i[i] - size_index[0])/ (size_index[1]-size_index[0]))
            all_TP_size_num[ind] = all_TP_size_num[ind] + 1

def TP_pred_recall_num(gt_rel_k, pred_to_gt_k):
    i_TP_pred_num = np.zeros([3, detector.num_rels])
    i_pred_num = np.zeros([detector.num_rels])
    for gt_rels_i in gt_rel_k:
        i_pred_num[gt_rels_i[2]] = i_pred_num[gt_rels_i[2]] + 1
        all_pred_num[gt_rels_i[2]] = all_pred_num[gt_rels_i[2]] + 1 

    for k in evaluator[conf.mode].result_dict[conf.mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt_k[:k])
        for j in match:
            j = int(j)
            if k==20:
                thr_k=0
            if k==50:
                thr_k=1
            if k==100:
                thr_k=2
            i_TP_pred_num[thr_k][gt_rel_k[j,2]] = i_TP_pred_num[thr_k][gt_rel_k[j,2]] + 1
            all_TP_pred_num[thr_k][gt_rel_k[j,2]] = all_TP_pred_num[thr_k][gt_rel_k[j,2]] + 1
    return i_TP_pred_num/(i_pred_num[None,:]+0.00001)

def list_rm_duplication(tri_list):
    old_size = tri_list.shape[0]
    all_rel_sets = defaultdict(list)
    for (o0, o1, r) in tri_list:
        all_rel_sets[(o0, o1)].append(r)
    gt_rels = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
    gt_rels = np.array(gt_rels)
    return gt_rels

def val_batch(batch_num, b, evaluator, evaluator_rel,thrs=(20, 50, 100)):

    det_res = detector[b]
    num_correct = 0
    num_sample = 0
    num_correct_adj_mat_rel = 0
    num_correct_adj_mat_obj = 0
    num_sample_adj_mat = 0
    TP_sample_obj = 0
    TP_sample_rel = 0
    True_sample = 0
    # the image size after resizing to IMAGE_SCALE (1024)
    if conf.num_gpus == 1:
        det_res = [det_res]

    if conf.mode != 'detclass':
        for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, \
                pred_adj_mat_rel_i, pred_adj_mat_obj_i, gt_adj_mat_i) in enumerate(det_res):
            gt_entry = {
                'gt_classes': val.gt_classes[batch_num + i].copy(),
                'gt_relations': val.relationships[batch_num + i].copy(),
                'gt_boxes': val.gt_boxes[batch_num + i].copy(),
                'gt_adj_mat': gt_adj_mat_i.copy(),
            }
            assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
            # assert np.all(rels_i[:,2] > 0)

            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,
                'pred_adj_mat_rel': pred_adj_mat_rel_i,
                'pred_adj_mat_obj': pred_adj_mat_obj_i,
            }
            all_pred_entries.append(pred_entry)
            num_sample = num_sample + objs_i.shape[0]
            num_sample_adj_mat = num_sample_adj_mat + gt_adj_mat_i.shape[0]
            True_sample = True_sample + (gt_adj_mat_i==1).sum()


            res_i = evaluator[conf.mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )
            res_i_rel = evaluator_rel[conf.mode].evaluate_scene_graph_entry(
                gt_entry,
                pred_entry,
            )
            gt_rel_k = val.relationships[batch_num + i].copy()
            pred_to_gt_k = res_i[0]
            all_pred_recall.append(TP_pred_recall_num(gt_rel_k, pred_to_gt_k))
                    
    else :
        for i, (boxes_i, objs_i, obj_scores_i) in enumerate(det_res):
            img_size = b.im_sizes[0][0]

            num_sample = num_sample + objs_i.shape[0]
            num_correct = num_correct + np.sum((val.gt_classes[batch_num + i].copy() - objs_i)==0)
            count_num(val.gt_classes[batch_num + i].copy())
            TP_count_num(val.gt_classes[batch_num + i].copy(), objs_i)
            count_size_num(val.gt_boxes[batch_num + i].copy() * (IM_SCALE / BOX_SCALE), img_size)
            TP_count_size_num(val.gt_classes[batch_num + i].copy(), objs_i,
                              val.gt_boxes[batch_num + i].copy() * (IM_SCALE / BOX_SCALE),
                              img_size)
    return num_correct, num_sample, num_correct_adj_mat_obj, num_correct_adj_mat_rel, num_sample_adj_mat, \
           True_sample, TP_sample_obj, TP_sample_rel



evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred)
evaluator_rel = BasicSceneGraphEvaluator_rel.all_modes(multiple_preds=conf.multi_pred)
if conf.cache is not None and os.path.exists(conf.cache):
    print("Found {}! Loading from it".format(conf.cache))
    with open(conf.cache,'rb') as f:
        all_pred_entries = pkl.load(f)
    conf_mat = np.zeros([detector.num_rels,detector.num_rels])
    for i, pred_entry in enumerate(tqdm(all_pred_entries)):
        gt_entry = {
            'gt_classes': val.gt_classes[i].copy(),
            'gt_relations': val.relationships[i].copy(),
            'gt_boxes': val.gt_boxes[i].copy(),
        }
        res_i = evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
        res_i_rel = evaluator_rel[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
        pred_prd_scores = pred_entry['rel_scores'][:, 1:]
        pred_prd_labels = pred_prd_scores.argmax(-1) + 1
        gt_labels_prd = val.relationships[i][:,2]
        det_boxes_sbj = pred_entry['pred_boxes'][pred_entry['pred_rel_inds'][:,0]]
        det_boxes_obj = pred_entry['pred_boxes'][pred_entry['pred_rel_inds'][:,1]]
        gt_boxes_sbj = val.gt_boxes[i][val.relationships[i][:,0]]
        gt_boxes_obj = val.gt_boxes[i][val.relationships[i][:,1]]
        for i in range(len(pred_prd_labels)):
            pred_prd_label = pred_prd_labels[i]
            det_boxes_sbj_i = det_boxes_sbj[i]
            det_boxes_sbj_i = det_boxes_sbj_i.astype(dtype=np.float32, copy=False)
            det_boxes_obj_i = det_boxes_obj[i]
            det_boxes_obj_i = det_boxes_obj_i.astype(dtype=np.float32, copy=False)
            gt_boxes_sbj_t = gt_boxes_sbj.astype(dtype=np.float32, copy=False)
            gt_boxes_obj_t = gt_boxes_obj.astype(dtype=np.float32, copy=False)
            sub_iou = bbox_overlaps(det_boxes_sbj_i[None, :4], gt_boxes_sbj[:,:4])[0]
            obj_iou = bbox_overlaps(det_boxes_obj_i[None, :4], gt_boxes_obj[:,:4])[0]
            inds = (sub_iou >= 0.5) & (obj_iou >= 0.5)
            max_iou = 0
            max_id = -1
            for j in range(len(inds)):
                if inds[j]:
                    if sub_iou[j] >= 0.5 and obj_iou[j] >= 0.5:
                        if sub_iou[j] * obj_iou[j] >= max_iou:
                            max_iou = sub_iou[j] * obj_iou[j]
                            max_id = j
            if max_id != -1:
                gt_prd_label = gt_labels_prd[max_id]
            else:
                gt_prd_label = 0
            conf_mat[gt_prd_label, pred_prd_label] = conf_mat[gt_prd_label, pred_prd_label] + 1
        gt_rel_k = val.relationships[i].copy()
        pred_to_gt_k = res_i[0]
        #all_pred_recall.append(TP_pred_recall_num(gt_rel_k, pred_to_gt_k))
    np.save('conf_mat.npy', conf_mat)
    evaluator[conf.mode].print_stats()
    evaluator_rel[conf.mode].print_stats()
    save_path = conf.ckpt.split('vgre')[0]
    file_name = conf.cache.split('/')[-1]
    np.save(save_path +'/'+ file_name + 'all_pred_recall.npy', np.array(all_pred_recall).mean(0))
    np.save(save_path +'/'+ file_name + 'all_pred_num.npy', all_pred_num)
    np.save(save_path +'/'+ file_name + 'all_TP_pred_num.npy', all_TP_pred_num)
else:
    detector.eval()
    num_correct = 0
    num_sample = 0
    num_correct_adj_rel = 0
    num_correct_adj_obj = 0
    num_sample_adj = 0
    True_sample = 0
    TP_sample_obj = 0
    TP_sample_rel = 0
    for val_b, batch in enumerate(tqdm(val_loader)):
        num_correct_i, num_sample_i, num_correct_adj_obj_i, num_correct_adj_rel_i, num_sample_adj_i, \
        True_sample_i, TP_sample_obj_i, TP_sample_rel_i  = val_batch(conf.num_gpus*val_b, batch,
                                                                     evaluator,evaluator_rel)
        num_correct = num_correct + num_correct_i
        num_sample = num_sample + num_sample_i
        num_correct_adj_rel = num_correct_adj_rel + num_correct_adj_rel_i
        num_correct_adj_obj = num_correct_adj_obj + num_correct_adj_obj_i
        num_sample_adj = num_sample_adj + num_sample_adj_i
        True_sample = True_sample + True_sample_i
        TP_sample_obj = TP_sample_obj + TP_sample_obj_i
        TP_sample_rel = TP_sample_rel + TP_sample_rel_i
    print('num_correct ',num_correct)
    print('num_sample',num_sample)
    print('obj acc:', (num_correct*1.0)/(num_sample*1.0))
    print('adj rel sum:', (num_correct_adj_rel * 1.0))
    print('adj obj sum:', (num_correct_adj_obj * 1.0))
    print('adj rel acc:', (num_correct_adj_rel * 1.0) / (num_sample_adj * 1.0))
    print('adj obj acc:', (num_correct_adj_obj * 1.0) / (num_sample_adj * 1.0))
    print('TP adj rel recall:', (TP_sample_rel * 1.0) / (True_sample * 1.0))
    print('TP adj obj recall:', (TP_sample_obj * 1.0) / (True_sample * 1.0))
    evaluator[conf.mode].print_stats()
    evaluator_rel[conf.mode].print_stats()
    if conf.cache is not None:
        with open(conf.cache,'wb') as f:
            pkl.dump(all_pred_entries, f)

    save_path = conf.ckpt.split('vgre')[0]
    file_name = conf.cache.split('/')[-1]
    np.save(save_path +'/'+ file_name + 'all_pred_recall.npy', np.array(all_pred_recall).mean(0))
    np.save(save_path +'/'+ file_name + 'all_pred_num.npy', all_pred_num)
    np.save(save_path +'/'+ file_name + 'all_TP_pred_num.npy', all_TP_pred_num)
    np.save(save_path + '/all_rel_num.npy', all_rel_num)
    np.save(save_path + '/all_TP_rel_rel_num.npy', all_TP_rel_rel_num)
    np.save(save_path + '/all_TP_rel_obj_num.npy', all_TP_rel_obj_num)
    np.save(save_path + '/label_recall.npy',all_TP_label_num / (1.0 * all_label_num))
    np.save(save_path + '/size_recall.npy', all_TP_size_num / (1.0 * all_size_num))
    np.save(save_path + '/all_TP_label_num.npy',all_TP_label_num)
    np.save(save_path + '/all_label_num.npy', all_label_num)
    np.save(save_path + '/all_TP_size_num.npy', all_TP_size_num)
    np.save(save_path + '/all_size_num.npy', all_size_num)



