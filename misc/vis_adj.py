"""
Visualization script. I used this to create the figures in the paper.
WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
#from lib.rel_model import RelModel
from lib.rel_model_topgcn import RelModel
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
from lib.fpn.box_utils import bbox_overlaps
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import os
from functools import reduce
import  pickle as pkl
conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)
if conf.test:
    val = test
entry = pkl.load(open('/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn/sgcls_gcn2_lstm2','rb'))
output_path = ''
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)
for i, ent_i in enumerate(tqdm(all_pred_entries)):
    adj_obj = ent_i['pred_adj_mat_obj']
    adj_rel = ent_i['pred_adj_mat_rel']
    obj_classes = ent_i['pred_classes']
    rel_ind = ent_i['pred_rel_inds']
    num_obj = obj_classes.shape[0]
    adj_mat_obj = np.zeros([num_obj,num_obj])
    adj_mat_rel = np.zeros([num_obj,num_obj])
    cont_i = 0
    img_name = val.filenames[i]
    for ind_i in num_obj:
        adj_mat_obj[ind_i[0]][ind_i[1]] = adj_obj[cont_i]
        adj_mat_rel[ind_i[0]][ind_i[1]] = adj_rel[cont_i]
        cont_i = cont_i + 1
    