# -*- coding: utf-8 -*-
"""
[1] Input shape [3,224,224]
"""
import torch
import torch.nn as nn

class my_net(nn.Module):
    def __init__(self):
        super(my_net, self).__init__()
        self.dropout_p = 0.5
        self.conv = nn.Sequential(         # input shape (4,10)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=10,             # n_filters
                kernel_size=4,              # filter size
                stride=1,                   # filter movement/step
            ),                              # output shape (10,6) 
            nn.ReLU(),
        )
        self.fl1 = nn.Sequential( 
            nn.Linear(10*6, 256),          # fully connected layer, output 10 classes
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fl2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fl3 = nn.Sequential(
            nn.Linear(128,5),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fl1(x)
        x = self.fl2(x)
        output = self.fl3(x)
        return output
