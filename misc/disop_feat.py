import numpy as np


f = '/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/edge_feat.npy'
feat = np.load(f)
sample_num = np.zeros(feat.shape[0])
for i in range(feat.shape[0]):
    for j in range(feat[i].shape[0]):
        if np.sum(feat[i][j] == 0) != 0:
            sample_num[i] += 1
print('max: ',sample_num.max(), 'min: ', sample_num.min())