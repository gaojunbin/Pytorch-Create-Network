# -*- coding: utf-8 -*-
"""
Created on 2020.6
Latest modify 2020.7
@Author: Junbin
@Note  : vgg net(modified)
"""
import torch
import torch.nn as nn

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.dropout_p = 0.5
        self.conv1 = nn.Sequential(         # input shape (3,224,224)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=4,             # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (4,224,224)
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=4,              # input height
                out_channels=8,             # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),    
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=8,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),    
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              # input height
                out_channels=32,             # n_filters
                kernel_size=3,               # filter size
                stride=1,                    # filter movement/step
                padding=1,                   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),  
        )
        self.conv5 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              # input height
                out_channels=64,             # n_filters
                kernel_size=3,               # filter size
                stride=1,                    # filter movement/step
                padding=1,                   # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),    
        )
        self.fl1 = nn.Sequential( 
            nn.Linear(64*7*7, 256),          # fully connected layer, output 10 classes
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fl2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.fl3 = nn.Sequential(
            nn.Linear(128,2),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fl1(x)
        x = self.fl2(x)
        output = self.fl3(x)
        return output
