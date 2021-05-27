import numpy as np
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
filename = 'conf_matrix.pdf'
figsize = [120,100]
prd_dist = np.load('/home/guoyuyu/guoyuyu/code/code_by_myself/scene_graph/dataset_analysis/rel_dis.npy')
prd_dist[0] = np.inf
VG_dict = json.load(open('/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
conf_mat = np.load('/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn/conf_mat.npy')
pred2ind  = VG_dict['predicate_to_idx']
prd_dict = ['']*(len(pred2ind)+1)
for pred_i in pred2ind.keys():
    prd_dict[pred2ind[pred_i]] = pred_i
prd_dict[0]='NULL'
prd_dict = np.array(prd_dict)
cm_sum = np.sum(conf_mat, axis=1, keepdims=True)
conf_mat = conf_mat / (cm_sum.astype(float)+1e-8)*100
prd_dist_sort = (0-prd_dist).argsort()
conf_mat_sort = conf_mat[prd_dist_sort,:]
conf_mat_sort = conf_mat_sort[:,prd_dist_sort]
prd_dist = prd_dist[prd_dist_sort]
prd_dict = prd_dict[prd_dist_sort]

cm = conf_mat_sort

cm_perc = cm #/ cm_sum.astype(float)*100
annot = np.empty_like(cm).astype(str)
nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = cm[i, j]
        p = cm_perc[i, j]
        s = cm_sum[i]
        annot[i, j] = '%.1f%%\n%.1f/%d' % (p, c, s)

cm = pd.DataFrame(cm, index=prd_dict, columns=prd_dict)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=figsize)
sns.heatmap(cm, annot=annot, fmt='', ax=ax)
plt.savefig(filename)