# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 10:50:06 2018

@author: billg_000
"""
import torch
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
        self.num_conv_outputs = 0
        
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

    def calculate_conv_layer_output_size(self, layers):
        """
        Calculate the number of outputs from the initial convolutional layers in the net
        :param layers: The layers at the start of the network (usually convolutional +
        maxpooling) whose output size is needed for the next layer
        :return: Integer -- number of outputs
        """
        # Calculate the size of the output from the conv layers passed in
        trial_out = torch.randn(1, 1, self.image_size[0], self.image_size[1])

        for layer in layers:
            trial_out = layer.forward(trial_out)
            
        self.num_conv_outputs = len(trial_out.flatten())
        print("Calculating outputs", self.num_conv_outputs)
        return self.num_conv_outputs
    
    
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


class ConvNet32(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet32, self).__init__(num_classes, channels, image_size)
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

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out


class ConvNet48(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet48, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=5, padding=2),  # (32 * prod(image_size))
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, padding=2),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out

class ConvNet32_753(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet32_753, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out

class ConvNet48_333(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet48_333, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # (32 * prod(image_size))
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out

class ConvNet48_Dropout(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet48_Dropout, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # (32 * prod(image_size))
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc1 = nn.Linear(self.num_conv_outputs, 500)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        return out

class ConvNet48_Dropout2(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet48_Dropout2, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # (32 * prod(image_size))
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc1 = nn.Linear(self.num_conv_outputs, 800)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(800, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        return out

class ConvNet48_Dropout3(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet48_Dropout3, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # (32 * prod(image_size))
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.drop1(out)
        out = self.fc1(out)

        return out

class ConvNet64(BaseNet):
    def __init__(self, num_classes, channels, image_size):
        super(ConvNet64, self).__init__(num_classes, channels, image_size)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),  # (32 * prod(image_size))
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                             # ( 32 * prod(image_size/2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, padding=2),  # ( 64 * prod(image_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))                              # ( 64 * prod(image_size/4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.calculate_conv_layer_output_size((self.layer1, self.layer2, self.layer3))

        self.fc = nn.Linear(self.num_conv_outputs, self.num_classes)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out