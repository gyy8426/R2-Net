import numpy as np
import h5py
import json
train_dataset = h5py.File("/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG.h5",'r')
father_dataset = h5py.File("/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG_father.h5",'w')
child_dataset = h5py.File("/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG_child.h5",'w')
pred_forest = json.load(open('/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/predicate_forest.json','r'))
vg_dict = json.load(open('/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
dict_ind2word = vg_dict['predicate_to_idx']
father_pred = pred_forest.keys()
child_pred = []
for fa_i in father_pred:
    child_pred.append(father_pred[fa_i])
def from_word2ind(word, dict_ind2word):
    ind = []
    for word_i in word:
        ind.append(word_dict[word_i])
father_ind = from_word2ind(father_pred, dict_ind2word)
child_ind = from_word2ind(child_pred, dict_ind2word)

def create_split_dataset(all_dataset, chose_id)
    