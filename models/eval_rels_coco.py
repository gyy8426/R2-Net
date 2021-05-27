import matplotlib
matplotlib.use('Agg')
from dataloaders.visual_genome import VGDataLoader, VG
from dataloaders.mscoco import CocoDetection as Coco
from dataloaders.mscoco import CocoDataLoader
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
top_num = 32
save_path = '/mnt/data/guoyuyu/datasets/mscoco/features/rel_features_gtbox/'
from lib.pytorch_misc import load_reslayer4
size_index = np.load('/home/guoyuyu/code/code_by_myself/scene_graph/dataset_analysis/size_index.npy')
conf = ModelConfig()
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
    #from lib.rel_model_best import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
else:
    raise ValueError()

train_vg, val_vg, test_vg = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

train_coco, val_coco = Coco.splits()

train_loader, val_loader = CocoDataLoader.splits(train_coco, val_coco, mode='det',
                                               batch_size=1,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               shuffle=False)

detector = RelModel(classes=train_vg.ind_to_classes, rel_classes=train_vg.ind_to_predicates,
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
                    ext_feat=True,
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



def val_batch(batch_num, b):
    det_res = detector[b]
    num_correct = 0
    num_sample = 0
    num_correct_adj_mat = 0
    num_sample_adj_mat = 0
    # the image size after resizing to IMAGE_SCALE (1024)
    if conf.num_gpus == 1:
        det_res = [det_res]
    for i, (id_i, boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, pred_adj_mat_i, \
            pred_adj_mat_1_i, gt_adj_mat_i, feat_i) in enumerate(det_res):

        if feat_i.shape[0] > top_num:
            get_num = top_num
        else:
            get_num = feat_i.shape[0]
        feat_i_t = feat_i[:get_num]
        rels_i_t = rels_i[:get_num]
        sub_i = objs_i[rels_i_t[:,0]]
        obj_i = objs_i[rels_i_t[:,1]]
        rels_pred_i = 1 + pred_scores_i[:get_num,1:].argmax(1)
        rels = []
        for k in range(sub_i.shape[0]):
            rel_k = []
            rel_k.append(train_vg.ind_to_classes[sub_i[k]])
            rel_k.append(train_vg.ind_to_predicates[rels_pred_i[k]])
            rel_k.append(train_vg.ind_to_classes[obj_i[k]])
            rels.append(rel_k)
        id_i = id_i[0][0]
        #print('rels: ', rels)
        #print('id_i: ', id_i)
        #print('file name: ', train_coco.coco.loadImgs(int(id_i)))
        np.savez(save_path+str(id_i)+'.npz', feat=feat_i_t, \
                 rel= np.column_stack((sub_i, rels_pred_i, obj_i)),\
                 rel_word=rels)
        #torch.cuda.empty_cache()


    return num_correct, num_sample, num_correct_adj_mat, num_sample_adj_mat,


detector.eval()

loaders = [train_loader, val_loader]
for val_b, batch in enumerate(tqdm(train_loader)):
    num_correct_i, num_sample_i, num_correct_adj_i, num_sample_adj_i \
        = val_batch(conf.num_gpus*val_b, batch)
torch.cuda.empty_cache()
for val_b, batch in enumerate(tqdm(val_loader)):
    num_correct_i, num_sample_i, num_correct_adj_i, num_sample_adj_i \
        = val_batch(conf.num_gpus*val_b, batch)
torch.cuda.empty_cache()






