import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv, SumPooling
from .base_model import BaseModel

class GINModel(BaseModel, nn.Module): # 继承 BaseModel 和 nn.Module
    """Simplified Graph Isomorphism Network."""

    def __init__(self, input_size=1, num_classes=2):
        super().__init__()
        self.conv1 = GINConv(
            nn.Linear(input_size, num_classes), aggregator_type="sum"
        )
        self.conv2 = GINConv(
            nn.Linear(num_classes, num_classes), aggregator_type="sum"
        )
        self.pool = SumPooling()

    def forward(self, g, feats):
        feats = self.conv1(g, feats)
        feats = F.relu(feats)
        feats = self.conv2(g, feats)
        return self.pool(g, feats)