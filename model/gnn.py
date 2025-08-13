# -*- coding: utf-8 -*-
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import numpy as np
from . import config

class GCNLayer(nn.Module):
    """GCN Module layer.
    """

    def __init__(self, args):
        super(GCNLayer, self).__init__()
        self.args = args
        self.dropout = config.dropout_rate
        self.layer_number = config.num_gnn_layers

        self.encoder = []
        for i in range(self.layer_number):
            self.encoder.append(GNN(
                nfeat=config.hidden_size,
                nhid=config.hidden_size,
                dropout=config.dropout_rate,
                alpha=config.leakey))

    def forward(self, fea, adj):
        learn_fea = fea
        tmp_fea = fea
        for layer in self.encoder:
            learn_fea = F.dropout(learn_fea, self.dropout,
                                  training=self.training)
            learn_fea = layer(learn_fea, adj)  # GNN.forward
            tmp_fea = tmp_fea + learn_fea
        return tmp_fea / (self.layer_number + 1)


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GNN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        return x


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features).to("cuda:0"))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * \
                  2 * init_range - init_range
        return nn.Parameter(initial / 2)

    def forward(self, input, adj):
        support = input
        output = torch.spmm(adj, support) #(num_item,num_item)*(num_item,emb_size)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + " (" \
               + str(self.in_features) + " -> " \
               + str(self.out_features) + ")"
