from .epoch_level_encoder import Epoch_Level_Encoder
from .seq_level_encoder import Seq_level_Encoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepLT(nn.Module):
    def __init__(self, num_classes=5, mode=None, epoch_num=20):
        super(SleepLT, self).__init__()
        self.epoch_encoder = Epoch_Level_Encoder()
        self.seq_encoder = Seq_level_Encoder(
            num_heads=4,
            hidden_dim=128,
            mlp_dim=128,
            dropout=0.1,
            attention_dropout=0.1,
        )
        self.classifier = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.mode = mode

        self.out_feature = False
    
    def set_out_feature(self, out_feature):
        self.out_feature = out_feature
    
    def forward(self, x):
        batch_size, num_epochs, num_channels, num_samples = x.shape
        x = x.view(batch_size*num_epochs, 1, -1)
        
        x = self.epoch_encoder(x)
        x = x.view(batch_size, num_epochs, -1)
        
        _y = x
        x = self.seq_encoder(x)

        y = self.classifier(x)
        if self.out_feature:
            return _y.view(batch_size*num_epochs, -1), y.view(batch_size*num_epochs, -1)
        return y
    
    def cal_params(self):
        params = 0
        for param in self.parameters():
            params += param.numel()
        print(f"Total params: {params}")
        return params
