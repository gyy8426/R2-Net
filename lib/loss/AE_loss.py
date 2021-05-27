import torch
import time
import numpy as np
from torch.autograd import Variable
from lib.pytorch_misc import make_input, convert_symmat

def singleTagLoss(pred_tag, keypoints):
    """
    associative embedding loss for one image
    """
    eps = 1e-6
    tags = []
    pull = 0
    for i in keypoints:
        tmp = []
        for j in i:
            if j[1]>0:
                tmp.append(pred_tag[j[0]])
        if len(tmp) == 0:
            continue
        tmp = torch.stack(tmp)
        tags.append(torch.mean(tmp, dim=0))
        pull = pull +  torch.mean((tmp - tags[-1].expand_as(tmp))**2)

    if len(tags) == 0:
        return make_input(torch.zeros([1]).float()), make_input(torch.zeros([1]).float())

    tags = torch.stack(tags)[:,0]

    num = tags.size()[0]
    size = (num, num, tags.size()[1])
    A = tags.unsqueeze(dim=1).expand(*size)
    B = A.permute(1, 0, 2)

    diff = A - B
    diff = torch.pow(diff, 2).sum(dim=2)[:,:,0]
    push = torch.exp(-diff)
    push = (torch.sum(push) - num)
    return push/((num - 1) * num + eps) * 0.5, pull/(num + eps)

def tagLoss(tags, keypoints):
    """
    accumulate the tag loss for each image in the batch
    """
    pushes, pulls = [], []
    keypoints = keypoints.cpu().data.numpy()
    for i in range(tags.size()[0]):
        push, pull = singleTagLoss(tags[i], keypoints[i%len(keypoints)])
        pushes.append(push)
        pulls.append(pull)
    return torch.stack(pushes), torch.stack(pulls)

def AE_loss(ass_embed, adj_mat, result):
    """

    Args:
        ass_embed: num_objs, dim_embed
        adj_mat: num_objs, num_objs

    Returns:

    """
    spare_adj_mat = torch.zeros([adj_mat.size(0), adj_mat.size(0)]).cuda(ass_embed.get_device(), async=True)
    spare_adj_mat = Variable(spare_adj_mat)
    spare_adj_mat[result.rel_labels_graph[:, 1], result.rel_labels_graph[:, 2]]\
        = adj_mat[result.rel_labels_graph[:, 1], result.rel_labels_offset_graph[:, 2]].type_as(ass_embed.data)
    adj_mat = convert_symmat(spare_adj_mat)
    im_mat = (result.im_inds[:, None] == result.im_inds[None, :]).type_as(ass_embed.data)
    eye_mat = torch.eye(adj_mat.size(0)).cuda(adj_mat.get_device(),async=True)
    eye_mat = Variable(eye_mat)
    im_mat = im_mat - eye_mat
    adj_mat_f = im_mat - adj_mat
    #print('result.gt_adj_mat_graph',result.gt_adj_mat_graph)
    #print('adj_mat',adj_mat)
    #print('adj_mat_f', adj_mat_f)
    eps = 1e-6
    num = ass_embed.size()[0]
    size = (num, num, ass_embed.size()[1])
    ass_embed_a = ass_embed.unsqueeze(dim=1).expand(*size)
    ass_embed_b = ass_embed_a.permute(1, 0, 2)
    diff_push = adj_mat_f[:, :, None] * (ass_embed_a - ass_embed_b)
    #print('diff_push: ',diff_push)
    diff_push = torch.pow(diff_push, 2).sum(dim=2)
    push = adj_mat_f * torch.exp(-diff_push)
    #push = torch.sum(push/(adj_mat_f.sum(dim=1)[:,None] + eps))
    #push = push / num * 0.5
    push = torch.sum(push) / (adj_mat_f.sum() + eps) * 0.5

    diff_pull = adj_mat[:, :, None] * (ass_embed_a - ass_embed_b)
    diff_pull = adj_mat * torch.pow(diff_pull, 2).sum(dim=2)
    #pull =  torch.sum(diff_pull/(adj_mat.sum(dim=1)[:,None] + eps))
    #pull = pull / num
    pull = torch.sum(diff_pull) / (adj_mat.sum()+ eps)
    return (push + pull)