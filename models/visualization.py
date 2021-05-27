"""
Visualization script. I used this to create the figures in the paper.
WARNING: I haven't tested this in a while. It's possible that some later features I added break things here, but hopefully there should be easy fixes. I'm uploading this in the off chance it might help someone. If you get it to work, let me know (and also send a PR with bugs/etc)
"""

from dataloaders.visual_genome import VGDataLoader, VG
from lib.rel_model import RelModel
#from lib.rel_model_topgcn import RelModel
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
from config import DATA_PATH
import scipy.misc
conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
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
                    )
detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])


############################################ HELPER FUNCTIONS ###################################

def get_cmap(N):
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index)) * (255 - pad) + pad)

    return map_index_to_rgb_color


cmap = get_cmap(len(train.ind_to_classes) + 1)


def load_unscaled(fn):
    """ Loads and scales images so that it's 1024 max-dimension"""
    image_unpadded = Image.open(fn).convert('RGB')
    im_scale = 1024.0 / max(image_unpadded.size)

    image = image_unpadded.resize((int(im_scale * image_unpadded.size[0]), int(im_scale * image_unpadded.size[1])),
                                  resample=Image.BICUBIC)
    return image


font = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf', 32)


def draw_box(draw, boxx, cls_ind, text_str):
    box = tuple([float(b) for b in boxx])
    if '-GT' in text_str:
        color = (255, 128, 0, 255)
    else:
        color = (0, 128, 0, 255)

    # color = tuple([int(x) for x in cmap(cls_ind)])

    # draw the fucking box
    draw.line([(box[0], box[1]), (box[2], box[1])], fill=color, width=8)
    draw.line([(box[2], box[1]), (box[2], box[3])], fill=color, width=8)
    draw.line([(box[2], box[3]), (box[0], box[3])], fill=color, width=8)
    draw.line([(box[0], box[3]), (box[0], box[1])], fill=color, width=8)

    # draw.rectangle(box, outline=color)
    w, h = draw.textsize(text_str, font=font)

    x1text = box[0]
    y1text = max(box[1] - h, 0)
    x2text = min(x1text + w, draw.im.size[0])
    y2text = y1text + h
    #print("drawing {}x{} rectangle at {:.1f} {:.1f} {:.1f} {:.1f}".format(
    #    h, w, x1text, y1text, x2text, y2text))

    draw.rectangle((x1text, y1text, x2text, y2text), fill=color)
    draw.text((x1text, y1text), text_str, fill='black', font=font)
    return draw

def find_rel_ind(rel_ind,sub_ind, obj_ind):
    for i, rel_i in enumerate(rel_ind):
        if rel_i[0] == sub_ind and rel_i[1] == obj_ind:
            return i
    return None

def generate_adj_mat(boxes, box_name, rel_score, rel_ind ):
    if len(box_name) == 0:
        return None, None
    adj_mat = np.zeros([len(box_name),len(box_name)])
    con_i = 0
    adj_ind_dict = {}
    adj_ind_name = ''
    for i in (box_name.keys()):
        adj_ind_dict[i] = con_i
        adj_ind_name = adj_ind_name+ 'ind '+ str(con_i) +': '+ box_name[i] + '|'
        con_i = con_i + 1
    adj_max = 0.0
    adj_min = 3.0
    for i, rel_ind_i in enumerate(box_name.keys()):
        for j, rel_ind_j in enumerate(box_name.keys()):
            if rel_ind_j != rel_ind_i:
                score = rel_score[find_rel_ind(rel_ind, rel_ind_i, rel_ind_j)]
                adj_mat[adj_ind_dict[rel_ind_i],adj_ind_dict[rel_ind_j]] = score
                if score > adj_max:
                    adj_max = score
                if score < adj_min:
                    adj_min = score
    adj_mat = (adj_mat-adj_min)/(adj_max-adj_min+1e-8)
    adj_mat = np.kron(adj_mat, np.ones((100, 100))) # Kronecker product for resize
    return adj_mat, adj_ind_name

def val_epoch():
    evaluator = BasicSceneGraphEvaluator.all_modes()
    if conf.cache is None or not os.path.exists(conf.cache):
        detector.eval()
        for val_b, batch in enumerate(tqdm(val_loader)):
            val_batch(conf.num_gpus * val_b, batch, evaluator)
    else:
        with open(conf.cache, 'rb') as f:
            all_pred_entries = pkl.load(f)
        for i, pred_entry in enumerate(tqdm(all_pred_entries)):
            """
            ['ids', 'pred_boxes', 'pred_classes', \
             'pred_rel_inds', 'obj_scores', 'rel_scores', \
             'pred_adj_mat_rel',
             'pred_adj_mat_obj']
            """
            det_res = (pred_entry['pred_boxes'],pred_entry['pred_classes'],
                       pred_entry['obj_scores'],pred_entry['pred_rel_inds'],
                       pred_entry['rel_scores'],pred_entry['pred_adj_mat_rel'],
                       pred_entry['pred_adj_mat_obj'])
            val_batch(batch_num = i, b=None, evaluator=evaluator,
                      det_res=det_res)
    evaluator[conf.mode].print_stats()


def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100), det_res=None):

    # if conf.num_gpus == 1:
    #     det_res = [det_res]
    assert conf.num_gpus == 1

    if conf.cache is None or not os.path.exists(conf.cache):
        det_res = detector[b]
        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, \
        pred_adj_mat_rel_i, pred_adj_mat_obj_i, _ = det_res
    else:
        boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, \
        pred_adj_mat_rel_i, pred_adj_mat_obj_i = det_res
    gt_entry = {
        'gt_classes': val.gt_classes[batch_num].copy(),
        'gt_relations': val.relationships[batch_num].copy(),
        'gt_boxes': val.gt_boxes[batch_num].copy(),
    }
    # gt_entry = {'gt_classes': gtc[i], 'gt_relations': gtr[i], 'gt_boxes': gtb[i]}
    assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
    # assert np.all(rels_i[:,2] > 0)
    if conf.cache is None or not os.path.exists(conf.cache):
        boxes_i = boxes_i * BOX_SCALE/IM_SCALE

    pred_entry = {
        'pred_boxes': boxes_i,
        'pred_classes': objs_i,
        'pred_rel_inds': rels_i,
        'obj_scores': obj_scores_i,
        'rel_scores': pred_scores_i,
        'pred_adj_mat_rel': pred_adj_mat_rel_i,
        'pred_adj_mat_obj': pred_adj_mat_obj_i,
    }
    pred_to_gt, pred_5ples, rel_scores = evaluator[conf.mode].evaluate_scene_graph_entry(
        gt_entry,
        pred_entry,
    )

    # SET RECALL THRESHOLD HERE
    pred_to_gt = pred_to_gt[:20]
    pred_5ples = pred_5ples[:20]

    # Get a list of objects that match, and GT objects that dont
    objs_match = (bbox_overlaps(pred_entry['pred_boxes'], gt_entry['gt_boxes']) >= 0.5) & (
            objs_i[:, None] == gt_entry['gt_classes'][None]
    )
    objs_matched = objs_match.any(1)

    has_seen = defaultdict(int)
    has_seen_gt = defaultdict(int)
    pred_ind2name = {}
    gt_ind2name = {}
    edges = {}
    missededges = {}
    badedges = {}
    edges_adj = {}
    obj_edges_adj = {}
    missededges_adj = {}
    obj_missededges_adj = {}
    badedges_adj = {}
    obj_badedges_adj = {}
    if val.filenames[batch_num].startswith('2343676'):
        import ipdb
        ipdb.set_trace()

    def query_pred(pred_ind):
        if pred_ind not in pred_ind2name:
            has_seen[objs_i[pred_ind]] += 1
            pred_ind2name[pred_ind] = '{}-{}'.format(train.ind_to_classes[objs_i[pred_ind]],
                                                     has_seen[objs_i[pred_ind]])
        return pred_ind2name[pred_ind]

    def query_relind(head,tail):
        for i in range(len(rels_i)):
            if rels_i[i][0]==head and rels_i[i][1] == tail:
                return i
        return None

    def query_gt(gt_ind):
        gt_cls = gt_entry['gt_classes'][gt_ind]
        if gt_ind not in gt_ind2name:
            has_seen_gt[gt_cls] += 1
            gt_ind2name[gt_ind] = '{}-GT{}'.format(train.ind_to_classes[gt_cls], has_seen_gt[gt_cls])
        return gt_ind2name[gt_ind]

    matching_pred5ples = pred_5ples[np.array([len(x) > 0 for x in pred_to_gt])]
    matching_ind = []
    for fiveple in matching_pred5ples:
        head_name = query_pred(fiveple[0])
        tail_name = query_pred(fiveple[1])
        edges[(head_name, tail_name)] = train.ind_to_predicates[fiveple[4]]
        relind = query_relind(fiveple[0], fiveple[1])
        matching_ind.append(relind)
        edges_adj[(head_name, tail_name)] = pred_adj_mat_rel_i[relind]
        obj_edges_adj[(head_name, tail_name)] = pred_adj_mat_obj_i[relind]
    matching_ind = np.array(matching_ind)
    gt_5ples = np.column_stack((gt_entry['gt_relations'][:, :2],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 0]],
                                gt_entry['gt_classes'][gt_entry['gt_relations'][:, 1]],
                                gt_entry['gt_relations'][:, 2],
                                ))
    has_match = reduce(np.union1d, pred_to_gt)
    for gt in gt_5ples[np.setdiff1d(np.arange(gt_5ples.shape[0]), has_match)]:
        # Head and tail
        namez = []
        pred_match = []
        for i in range(2):
            matching_obj = np.where(objs_match[:, gt[i]])[0]
            if matching_obj.size > 0:
                name = query_pred(matching_obj[0])
                pred_match.append(matching_obj[0])
            else:
                name = query_gt(gt[i])
            namez.append(name)

        missededges[tuple(namez)] = train.ind_to_predicates[gt[4]]
        if len(pred_match) == 2:
            relind = query_relind(pred_match[0], pred_match[1])
        else:
            relind = None

        if relind is not None:
            missededges_adj[tuple(namez)] = pred_adj_mat_rel_i[relind]
            obj_missededges_adj[tuple(namez)] = pred_adj_mat_obj_i[relind]
        else:
            missededges_adj[tuple(namez)] = 0.0
            obj_missededges_adj[tuple(namez)] = 0.0

    for fiveple in pred_5ples[np.setdiff1d(np.arange(pred_5ples.shape[0]), matching_ind)]:
        if fiveple[0] in pred_ind2name:
            if fiveple[1] in pred_ind2name:
                badedges[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = train.ind_to_predicates[fiveple[4]]
                relind = query_relind(fiveple[0],fiveple[1])
                badedges_adj[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = pred_adj_mat_rel_i[relind]
                obj_badedges_adj[(pred_ind2name[fiveple[0]], pred_ind2name[fiveple[1]])] = pred_adj_mat_obj_i[relind]
    theimg = load_unscaled(val.filenames[batch_num])
    theimg2 = theimg.copy()
    draw2 = ImageDraw.Draw(theimg2)

    # Fix the names

    for pred_ind in pred_ind2name.keys():
        draw2 = draw_box(draw2, pred_entry['pred_boxes'][pred_ind],
                         cls_ind=objs_i[pred_ind],
                         text_str=pred_ind2name[pred_ind])
    # for gt_ind in gt_ind2name.keys():
    #     draw2 = draw_box(draw2, gt_entry['gt_boxes'][gt_ind],
    #                      cls_ind=gt_entry['gt_classes'][gt_ind],
    #                      text_str=gt_ind2name[gt_ind])

    obj_rel_adj_mat, obj_rel_adj_ind_name  = generate_adj_mat(boxes=pred_entry['pred_boxes'],
                                       box_name=pred_ind2name,
                                       rel_score=pred_adj_mat_obj_i, rel_ind=rels_i)
    edge_rel_adj_mat, edge_rel_adj_ind_name = generate_adj_mat(boxes=pred_entry['pred_boxes'],
                                        box_name=pred_ind2name,
                                        rel_score=pred_adj_mat_rel_i,rel_ind=rels_i)

    recall = int(100 * len(reduce(np.union1d, pred_to_gt)) / gt_entry['gt_relations'].shape[0])
    id = '{}-{}'.format(val.filenames[batch_num].split('/')[-1][:-4], recall)
    pathname = os.path.join(DATA_PATH,'qualitative_sgcls_2adj', id)
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    theimg.save(os.path.join(pathname, 'img.jpg'), quality=100, subsampling=0)
    theimg2.save(os.path.join(pathname, 'imgbox.jpg'), quality=100, subsampling=0)
    if obj_rel_adj_mat is not None:
        scipy.misc.imsave(os.path.join(pathname, 'obj_rel_adj_mat.jpg'), obj_rel_adj_mat)
    if edge_rel_adj_mat is not None:
        scipy.misc.imsave(os.path.join(pathname, 'edge_rel_adj_mat.jpg'), edge_rel_adj_mat)
    with open(os.path.join(pathname, 'shit.txt'), 'w') as f:
        if obj_rel_adj_ind_name is not None:
            f.write('obj adj index name: {}\n'.format(obj_rel_adj_ind_name))
        if edge_rel_adj_ind_name is not None:
            f.write('edge adj index name: {}\n'.format(edge_rel_adj_ind_name))
        f.write('good:\n')
        for (o1, o2), p in edges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        for (o1, o2), p in edges_adj.items():
            f.write('rel {} - {} - {}\n'.format(o1, str(p), o2))
        for (o1, o2), p in obj_edges_adj.items():
            f.write('obj {} - {} - {}\n'.format(o1, str(p), o2))
        f.write('fn:\n')
        for (o1, o2), p in missededges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        for (o1, o2), p in missededges_adj.items():
            f.write('rel {} - {} - {}\n'.format(o1, str(p), o2))
        for (o1, o2), p in obj_missededges_adj.items():
            f.write('obj {} - {} - {}\n'.format(o1, str(p), o2))
        f.write('shit:\n')
        for (o1, o2), p in badedges.items():
            f.write('{} - {} - {}\n'.format(o1, p, o2))
        for (o1, o2), p in badedges_adj.items():
            f.write('rel {} - {} - {}\n'.format(o1, str(p), o2))
        for (o1, o2), p in obj_badedges_adj.items():
            f.write('obj {} - {} - {}\n'.format(o1, str(p), o2))


mAp = val_epoch()