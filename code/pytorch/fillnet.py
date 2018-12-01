"""
Implementation of a GRNN to do image infill using techniques based on
paper:
    
Application of GRNN neural network in non-texture image inpainting and restoration
Vahid K. Aliloua, Farzin Yaghmaeeb,∗

with modificications
"""
import torch
import torch.nn as nn
import numpy as np

class FillNet(nn.Module):
    
    def __init__(self, sigma=1.2):
        """
        Initialize a network which will be filled out as training
        samples are added
        """
        super(FillNet, self).__init__()
        self.sigma = sigma
        self.sigmaSq = torch.tensor(sigma**2, dtype=torch.float)
        self.reset()
        
    def reset(self):
        self.pattern_layer = None
        self.summation_layer = torch.zeros(2)
        self.traning_x_acc = []
        self.traing_target_acc = []
        self.W2 = None
        
    def add_one_training_sample(self, X, target):
        """
        x: a list [x1, x2] representing a co-ordinate in the image
        to use for training
        target: A value from 0 to 1 for the pixel value
        """
        self.traning_x_acc.append(X)
        self.traing_target_acc.append(target)
        
    def start_training(self):
        """
        Use accumulated training samples with their targets to train
        the model.
        """
        # Convert training data to torch tensors
        self.pattern_layer = torch.tensor(np.vstack(self.traning_x_acc))
        self.W2 = torch.randn(len(self.traning_x_acc), requires_grad=True)
        #self.W2 = torch.zeros(len(self.traning_x_acc), requires_grad=True)
        self.targets = torch.tensor(self.traing_target_acc)
        
    def rbf(self, W2, Dsquared):
        return W2 * torch.exp(-1 * Dsquared / (2*self.sigmaSq))
        
    def forward(self, X):
        self.X = X
        out = torch.zeros(len(X))
        
        for idx in range(len(X)):
            
            # Get distances to all pattern layer points from the input point
            self.diffs = (self.pattern_layer - X[idx])
            self.Dsquared = self.diffs.pow(2).sum(dim=1).type(torch.float)
            out[idx] = self.rbf(self.W2, self.Dsquared).sum() / self.rbf(1, self.Dsquared).sum()

        return out
    
    def parameters(self):
        """
        return a list of parameters
        """
        return [self.W2]
    
if __name__ == "__main__":
    ta = FillNet()
    
    ta.add_one_training_sample([2,4], 0.5)
    ta.add_one_training_sample([6,3], 0.2)
    ta.add_one_training_sample([3,6], 0.9)
    ta.train()
    print(ta.forward(torch.tensor([6,3],dtype=torch.int)))

