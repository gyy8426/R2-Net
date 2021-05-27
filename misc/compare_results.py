import numpy as np
import json
import dill as pkl
pred_dict = json.load(open('/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
pred_to_idx = pred_dict['predicate_to_idx']
idx_to_pred = pred_dict['idx_to_predicate']
all_pred_recall = pkl.load(open('./output/rel_cat_recall_final.npz','rb'))
child_pred_recall = pkl.load(open('./output/rel_cat_recall_child.npz','rb'))
forest = json.load(open('./misc/predicate_forest.json','r'))
print(forest.keys())
father_pred = []
for fa_idx in forest.keys():
    if fa_idx != '0':
        father_pred.append(idx_to_pred[fa_idx])
child_pred = []
for fa_idx in forest.keys():
    for child_idx in forest[fa_idx]:
        if child_idx != 0:
            child_pred.append(idx_to_pred[str(child_idx)])
def get_all_mean_recall(pred_recall):
    all_mean_recall = {}
    for k in pred_recall.keys():
        all_mean_recall[k] = []
        for pred_i in pred_recall[k].keys():
            if pred_i in child_pred:
                all_mean_recall[k].append(pred_recall[k][pred_i])
        print('top ',k,' :', np.mean(all_mean_recall[k]))
print('all_pred_recall: ')
get_all_mean_recall(all_pred_recall)
print('child_pred_recall: ')
get_all_mean_recall(child_pred_recall)        

