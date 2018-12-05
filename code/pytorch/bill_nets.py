# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:50:06 2018

@author: billg_000
"""
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


class BaseNet(nn.Module):
    """
    Base class for pytorch networks; supplies convience functions
    """
    def __init__(self, num_classes, channels, image_size):
        super(BaseNet, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size
        
    def get_total_parms(self):
        total_parms = 0
        # Find the total number of parameters being trained
        for t in self.state_dict().values():
            total_parms += int(np.prod(t.shape))
        return total_parms
    
    def get_transformer(self):
        """
        Make a torch transformation function that the network expect.
        """
        return transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(self.image_size),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,) * self.channels, (0.5,) * self.channels)])
    
# Define a model class. This uses purelin() as the second-layer
# transfer function
class FlatNet(BaseNet):
    def __init__(self, num_classes, channels, image_size, hidden_size):
        super(FlatNet, self).__init__(num_classes, channels, image_size)
        self.input_size = self.channels * self.image_size[0] * self.image_size[1]
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        out = x.view(-1, self.input_size)  # Flatten
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class ConvNet(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(800, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out
    
    