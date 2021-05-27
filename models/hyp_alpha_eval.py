import matplotlib
matplotlib.use('Agg')
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE, DATA_PATH
import dill as pkl
import os
import pickle as cpkl
from lib.pytorch_misc import load_reslayer4
#size_index = np.load('/home/guoyuyu/code/code_by_myself/scene_graph/dataset_analysis/size_index.npy')
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model_hyp import RelModel
    #from lib.rel_model_2bias_2_hyp import RelModel
    #from lib.rel_model_topgcn_hyp import RelModel
    #from lib.rel_model_rnn2_hyp import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, nl_adj=conf.nl_adj,
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

all_TP_label_num = np.zeros([detector.num_classes])
all_label_num = np.zeros([detector.num_classes])

all_pred_size = []

def count_num(label_i):
    for i in range(label_i.shape[0]):
        all_label_num[label_i[i]] = all_label_num[label_i[i]] + 1

def TP_count_num(label_i, pred_i):
    TP_labe_ind = ((label_i - pred_i) == 0)
    TP_labe = TP_labe_ind * pred_i
    for i in range(TP_labe.shape[0]):
        if TP_labe_ind[i]:
            all_TP_label_num[label_i[i]] = all_TP_label_num[label_i[i]] + 1


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
    det_res = detector[b]
    num_correct = 0
    num_sample = 0
    num_correct_adj_mat = 0
    num_sample_adj_mat = 0
    # the image size after resizing to IMAGE_SCALE (1024)
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (r) in enumerate(det_res):
        torch.cuda.empty_cache()
        for j in range(11):
            for k in range(11):
                boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, \
                pred_adj_mat_rel_i, pred_adj_mat_obj_i, gt_adj_mat_i = r[j][k]
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
                num_sample = num_sample + objs_i.shape[0]
                num_sample_adj_mat = num_sample_adj_mat + gt_adj_mat_i.shape[0]
                if conf.mode != 'sgdet':
                    num_correct = num_correct + np.sum((val.gt_classes[batch_num + i].copy() - objs_i)==0)
                    pred_adj_mat_bool_i = ( pred_adj_mat_rel_i > 0.5 ).astype('float64')
                    num_correct_adj_mat = num_correct_adj_mat + np.sum((gt_adj_mat_i - pred_adj_mat_bool_i)==0)
                evaluator[j][k][conf.mode].evaluate_scene_graph_entry(
                    gt_entry,
                    pred_entry,
                    alpha=0.1 * j, 
                    beta=0.1 * k,
                )

    return num_correct, num_sample, num_correct_adj_mat, num_sample_adj_mat,
evaluator = []
for i in range(11):
    evaluatori = []
    for j in range(11):
        evaluatori.append(BasicSceneGraphEvaluator.all_modes(multiple_preds=conf.multi_pred))
    evaluator.append(evaluatori)
detector.eval()
num_correct = 0
num_sample = 0
num_correct_adj = 0
num_sample_adj = 0
for val_b, batch in enumerate(tqdm(val_loader)):
    num_correct_i, num_sample_i, num_correct_adj_i, num_sample_adj_i = val_batch(conf.num_gpus*val_b, batch, evaluator)
    num_correct = num_correct + num_correct_i
    num_sample = num_sample + num_sample_i
    num_correct_adj = num_correct_adj + num_correct_adj_i
    num_sample_adj = num_sample_adj + num_sample_adj_i
print('num_correct ',num_correct)
print('num_sample',num_sample)
print('Amp:', (num_correct*1.0)/(num_sample*1.0))
print('adj Amp:', (num_correct_adj * 1.0) / (num_sample_adj * 1.0))
re = []
for i in range(11):
    print(i)
    re_i = []
    for j in range(11):
        print(j)
        evaluator[i][j][conf.mode].print_stats()
        re_i.append([evaluator[i][j][conf.mode].get_recall()])
    re.append(re_i)
import os
name_i = None
for i in range(100):
    name_t = 'val_alpha_beta'+'gcn2_lstm2_'+str(i)+'.pkl'
    if os.path.exists(name_t):
        continue
    else:
        name_i = name_t
        break
cpkl.dump(re, open(name_i,'wb'))

