import math

import torch
from torch import nn
from torch.nn import functional as F

class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat = True):
        super(GATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size = (in_features, out_features)))
        nn.init.kaiming_uniform_(self.W.data, fan = self.in_features, a = math.sqrt(5))

        self.a = nn.Parameter(torch.zeros(size = (out_features * 2, 1)))
        nn.init.kaiming_uniform_(self.a.data, fan = self.in_features, a = math.sqrt(5))

        self.LeakyRelu = nn.LeakyReLU(self.alpha)
    def forward(self, input, adj):
        # Linear Transformation
        h = torch.mm(input, self.W)
        N = h.size()[0]
        print(N)

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1,N).view(N * N, -1), h.repeat(N,1)], dim = 1).view(N, -1, 2 * self.out_features)
        e = self.LeakyRelu(torch.mm(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec = -9e15 * torch.zeros_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim = 1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.mm(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        