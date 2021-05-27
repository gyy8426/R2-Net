import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

class Complex_Embed(nn.Module):
    def __init__(self, input_dim, embed_dim, num_rel, dropout_rate):
        """

        Args:
            input_dim:
            embed_dim:
            num_rel:
            dropout_rate:
        """
        super(Complex_Embed, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_rel = num_rel

        self.sub_embed_re_layer = nn.Linear(self.input_dim, self.embed_dim)
        self.sub_embed_re_layer.weight = torch.nn.init.xavier_normal(self.sub_embed_re_layer.weight, gain=1.0)
        self.obj_embed_re_layer = nn.Linear(self.input_dim, self.embed_dim)
        self.obj_embed_re_layer.weight = torch.nn.init.xavier_normal(self.obj_embed_re_layer.weight, gain=1.0)

        self.sub_embed_im_layer = nn.Linear(self.input_dim, self.embed_dim)
        self.sub_embed_im_layer.weight = torch.nn.init.xavier_normal(self.sub_embed_im_layer.weight, gain=1.0)
        self.obj_embed_im_layer = nn.Linear(self.input_dim, self.embed_dim)
        self.obj_embed_im_layer.weight = torch.nn.init.xavier_normal(self.obj_embed_im_layer.weight, gain=1.0)

        self.rel_embed_re_layer = nn.Linear(self.embed_dim, self.num_rel, bias=False)
        self.rel_embed_re_layer.weight = torch.nn.init.xavier_normal(self.rel_embed_re_layer.weight, gain=1.0)
        self.rel_embed_im_layer = nn.Linear(self.embed_dim, self.num_rel, bias=False)
        self.rel_embed_im_layer.weight = torch.nn.init.xavier_normal(self.rel_embed_im_layer.weight, gain=1.0)

    @property
    def input_size(self):
        return self.input_dim

    def forward(self, obj_pre, union_feat, rel_ind):

        sub_re = self.sub_embed_re_layer(obj_pre)[rel_ind[:,1]] * union_feat
        sub_re = F.dropout(sub_re, self.dropout_rate, training=self.training)

        obj_re = self.obj_embed_re_layer(obj_pre)[rel_ind[:, 2]] * union_feat
        obj_re = F.dropout(obj_re, self.dropout_rate, training=self.training)

        sub_im = self.sub_embed_im_layer(obj_pre)[rel_ind[:, 1]] * union_feat
        sub_im = F.dropout(sub_im, self.dropout_rate, training=self.training)

        obj_im = self.obj_embed_im_layer(obj_pre)[rel_ind[:, 2]] * union_feat
        obj_im = F.dropout(obj_im, self.dropout_rate, training=self.training)

        rel = self.rel_embed_re_layer(sub_re * obj_re)   \
              + self.rel_embed_re_layer(sub_im * obj_im) \
              + self.rel_embed_im_layer(sub_re * obj_im) \
              - self.rel_embed_im_layer(sub_im * obj_re)

        return rel