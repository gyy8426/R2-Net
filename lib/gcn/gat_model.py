import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.gcn.pygat import GraphAttentionLayer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout, alpha=0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return x