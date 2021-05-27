import h5py
import numpy as np
import json
from collections import defaultdict
import matplotlib as mpl  
import matplotlib.pyplot as plt  
from scipy import stats 
import matplotlib.pylab as pylab
import seaborn as sns
from scipy import stats
import dill as pkl
rel_cate_recall = pkl.load(open('./output/rel_cat_recall.npz','rb'))

rel_cate_recall_vis = rel_cate_recall[100]
del rel_cate_recall_vis['all_rel_cates']
rel_cate_dist = np.load(open('./output/rel_dis.npy','rb'))
rel_cate_dist= rel_cate_dist[1:]
rel_dict = json.load(open('/mnt/data1/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
ind_rel = rel_dict['idx_to_predicate']
rel_ind = rel_dict['predicate_to_idx']

def dict2list(dic:dict,rel_cate_dist):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val, dist) for key, val, dist in zip(keys, vals, rel_cate_dist)]
    return lst

def draw_hist_from_dic(dict, name='None',step=5):
    fig_length = len(dict)
    params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '45',
        'ytick.labelsize': '20',
        'lines.linewidth': '8',
        'legend.fontsize': '25',
        'figure.figsize': str(fig_length)+', 50'  # set figure size
    }
    pylab.rcParams.update(params)
    x = np.arange(len(dict))
    x_labels = []
    y_values = []
    plt.title(name)
    for i in dict:
        y_values.append(i[2])
        x_labels.append(i[0])
    plt.bar(x, y_values)
    plt.xticks(x, x_labels, rotation='vertical', weight=200)
    plt.savefig('./misc/'+name+'.pdf', dpi=200)
    #plt.legend(loc='best')
    plt.close('all')
    return 0
    
rel_dis_dic = sorted(dict2list(rel_cate_recall_vis,rel_cate_dist), key=lambda x:x[2], reverse=True)
draw_hist_from_dic(rel_dis_dic,'dist_of_labels')