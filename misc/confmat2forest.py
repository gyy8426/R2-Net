import numpy as np
import json
prd_dist = np.load('/home/guoyuyu/guoyuyu/code/code_by_myself/scene_graph/dataset_analysis/rel_dis.npy')
prd_dist[0] = np.inf
VG_dict = json.load(open('/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
conf_mat = np.load('/home/guoyuyu/guoyuyu/code/code_by_other/neural-motifs_graphcnn/conf_mat.npy')
pred2ind  = VG_dict['predicate_to_idx']
prd_dict = ['']*(len(pred2ind)+1)
for pred_i in pred2ind.keys():
    prd_dict[pred2ind[pred_i]] = pred_i
prd_dict[0]='NULL'
def find_farther(vec, root_list):
    farther = np.argmax(vec)
    if farther not in root_list:
        print('father!')
        farther = find_farther(farther, root_list)
    else :
        return farther
        
def mat2forest(conf_mat):
    """
    confusion matrix into forest
    return 
    :param conf_mat: Number_Nodes * Number_Nodes
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    conf_mat_sum = np.sum(conf_mat, axis=1, keepdims=True)
    cm = conf_mat / (conf_mat_sum.astype(float)+1e-8)*100
    # conf_eye = conf_mat.diag()
    # prd_dict = np.array(prd_dict)
    # prd_dist_sort = (0 - conf_eye).argsort()
    # conf_mat_sort = conf_mat[prd_dist_sort,:]
    # conf_mat_sort = conf_mat_sort[:,prd_dist_sort]
    # prd_dist = prd_dist[prd_dist_sort]
    # prd_dict = prd_dict[prd_dist_sort]

    # for i in range(cm.shape[0]):
        # for j in range(cm.shape[1]):
            # if j > i :
                # cm[i,j] = -1
    forest = {}
    for i in range(cm.shape[0]):
        if cm[i,i] == max(cm[i,:]):
            forest[i] = []
    for i in range(cm.shape[0]):
        if i not in forest.keys():
            farther = find_farther(cm[i], forest.keys())
            forest[farther].append(i)
    return forest
            
if __name__ == '__main__':
    forest = mat2forest(conf_mat)
    for i in forest.keys():
        print('---------father:',prd_dict[i],'---------')
        print('children: ')
        for j in forest[i]:
            print(prd_dict[j],end=',')
        print('')
    with open('predicate_forest.json', 'w') as outfile:  
        json.dump(forest, outfile)