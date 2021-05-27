from PIL import Image, ImageDraw
#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import matplotlib
matplotlib.use('Agg')
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import cv2
import  config
import pickle as pkl
from dataloaders.visual_genome import load_graphs
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

split_mask, gt_boxes, gt_classes, gt_relationships = load_graphs(config.VG_SGG_FN, mode='test')
pred_data = pkl.loads(open('/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/top_gcn_onlyzeros.pkl'))
pred_score = pkl.loads(open('/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/test_cache_score.pkl'))
pred_score = pred_score['sgdet_recall'][20]
dict_data = json.load(open(config.VG_SGG_DICT_FN,'rb'))
obj_dict = dict_data['idx_to_label']
rel_dict = dict_data['idx_to_predicate']

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    pred_score_np = np.array(pred_score)
    pred_score_sort_ind = np.argsort(pred_score_np)
    for i in pred_score_sort_ind:
        image_name = pred_data[i]['ids']
        im = cv2.imread(image_name)
        pred_boxe_i = pred_data[i]['pred_boxes']
        pred_class_i = pred_data[i]['pred_classes']
        gt_boxes_i = gt_boxes[i]
        vis_detections(im,)
