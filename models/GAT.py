import torch

from torch import nn

from Layers.GATConv import GATConv
from GNN import GNN

class GAT(nn.Module):
    def __init__(self, num_features, hid_num, in_head, out_head):
        super(GAT, self).__init__()
        self.num_features = num_features
        self.hid_num = hid_num
        self.in_head = in_head
        self.out_head = out_head

        self.conv1 = GATConv(num_features, hid_num)

    def forward(self):
        pass
        