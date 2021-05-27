from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from collections import OrderedDict
from matplotlib import colors as mcolors
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns

from sklearn import preprocessing
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
f = '/home/guoyuyu/code/code_by_other/neural-motifs_graphcnn/misc/'
dictf = json.load(open('/mnt/data/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts.json','rb'))

type_o = 'edge'  #edge #obj
if type_o == 'obj':
    obj_dict = dictf['label_to_idx']
    #ind_dict = ['tree','table','bike','boy','shirt','banana','clock','laptop']
    ind_dict = ['dog', 'train', 'window',
                'arm', 'banana', 'man',
                'house' ,'flower', 'shirt',
                'sign']
    l_colors = [colors['red'], colors['blue'], colors['orange'],
                colors['green'], colors['black'], colors['deepskyblue'],
                colors['c'], colors['violet'], colors['m'],
                colors['maroon']
                ]
    '''
    ind_dict = obj_dict.keys()
    l_colors = []
    for i in colors.keys():
        l_colors.append(colors[i])
    '''
else:
    obj_dict = dictf['predicate_to_idx']
    no_zero = ['across against', 'attached to', 'growing on', 'standing on', 'to', 'watching']

    ind_dict = ['above','holding','wearing', \
                'looking at','riding','eating',\
                'lying on','on back of','with',
                'painted on']
    l_colors = [colors['red'], colors['blue'], colors['orange'],
                colors['green'], colors['black'], colors['deepskyblue'],
                colors['c'], colors['violet'], colors['m'],
                colors['maroon']
                ]
    '''

    ind_dict = obj_dict.keys()
    l_colors = []
    for i in colors.keys():
        l_colors.append(colors[i])
    '''
ind_label = []
for i in ind_dict:
    ind_label.append(obj_dict[i])
ind_label = np.array(ind_label)
def sigmoid(X):

    return 1.0 / (1 + np.exp(-float(X)));

num_samples = 300


def get_data():

    data = np.load(f+type_o+'_feat.npy')
    sp_ind = np.random.randint(0, data.shape[1]-100,size=(num_samples*ind_label.shape[0]))
    sp_ind = sp_ind.reshape([ind_label.shape[0], num_samples])
    data_s = np.zeros([ind_label.shape[0], num_samples, data.shape[-1]])
    for i in range(ind_label.shape[0]):
        data_s[i] = data[ind_label[i], sp_ind[i],:]
    label = np.array([i for i in range(data_s.shape[0]*data_s.shape[1])])
    label = (label / data_s.shape[1]).astype(int)
    data_s = data_s.reshape([-1, data_s.shape[-1]])
    n_features = data_s.shape[-1]
    n_samples_all = data_s.shape[0]
    return data_s, label, n_samples_all, n_features


def plot_embedding(data, label, title):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    '''
    data = preprocessing.scale(data)
    '''
    fig = plt.figure()
    ax = plt.subplot(111)
    #ax = plt.subplot(111, projection='3d')
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=l_colors[label[i]],
                 label=ind_dict[label[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend(ind_dict)
    return fig


def scatter(x, label):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    colors = []
    label_list = []
    color_float = []
    for i in range(len(ind_dict)):
        color_float.append(mcolors.to_rgba(l_colors[i])[:3])

    for i in range(len(label)):
        colors.append(color_float[label[i]])
        label_list.append(ind_dict[label[i]])

    for i in range(len(ind_dict)):
        start_ind = i * num_samples
        end_ind = (i+1) * num_samples
        sc = ax.scatter(x[start_ind:end_ind, 0], x[start_ind:end_ind, 1], lw=0, s=40,
                        c=color_float[i],
                        label=ind_dict[i])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    ax.legend()
    # We add the labels for each digit.
    txts = []
    '''
    for i in range(len(ind_dict)):
        # Position of each label.
        xtext, ytext = np.median(x[label == i, :], axis=0)
        txt = ax.text(xtext, ytext, ind_dict[i], fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    '''
    return f, ax, sc, txts




def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=201811, learning_rate=100) #,
    t0 = time()
    result = tsne.fit_transform(data)
    #fig = plot_embedding(result, label,
    #                     't-SNE embedding of object features')
    scatter(result,label)
    plt.savefig('fig_'+type_o+'_feat.pdf',format='pdf')


if __name__ == '__main__':
    main()