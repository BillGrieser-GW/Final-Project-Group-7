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
    
    def __init__(self, sigma=2.2):
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
        self.pattern_node_set = set()
        self.pattern_node_list = []
        self.W2 = None
        
    def add_one_pattern_node(self, X):
        """
        X: a vector [x1, x2] representing a co-ordinate in the image
        to use for training
        """
        X_tuple = (X[0], X[1])
        
        # Store if this is a new point
        if X_tuple not in self.pattern_node_set:
            self.pattern_node_set.add(X_tuple)
            self.pattern_node_list.append(X_tuple)
        
    def start_training(self):
        """
        Prepare the model so it can be trained.
        """
        # Convert training data to torch tensors and build the pattern layer
        self.pattern_layer = torch.tensor(np.vstack(self.pattern_node_list))
        
        # Initialize the weights
        #self.W2 = torch.rand(len(self.pattern_node_list), requires_grad=True)
        self.W2 = torch.full((len(self.pattern_node_list),), 0.1, requires_grad=True)
        
    def rbf(self, W2, Dsquared):
        return W2 * (torch.exp(-1 * Dsquared / (2*self.sigmaSq)))
        
    def forward(self, X):
        self.X = X
        out = torch.zeros(len(X))
        
#        for idx in range(len(X)):
#            
#            # Get sqaured distances to all pattern layer points from the input point
#            self.diffs = (self.pattern_layer - X[idx])
#            self.Dsquared = self.diffs.pow(2).sum(dim=1).type(torch.float)
#            
#            out[idx] = self.rbf(self.W2, self.Dsquared).sum() / self.rbf(1, self.Dsquared).sum()

        self.Dsquared = torch.zeros((len(X), len(self.pattern_layer)))
        
        for idx in range(len(X)):
            
           # Get sqaured distances to all pattern layer points from this X
           self.Dsquared[idx] = (self.pattern_layer - X[idx]).pow(2).sum(dim=1).type(torch.float)
           
        out = self.rbf(self.W2, self.Dsquared).sum(dim=1) / self.rbf(1, self.Dsquared).sum(dim=1)
        #out = self.rbf(self.W2, self.Dsquared).sum(dim=1) 
        
        return out
    
    def parameters(self):
        """
        return a list of parameters
        """
        return [self.W2]
    
if __name__ == "__main__":
    ta = FillNet()
    
    ta.add_one_pattern_node([2,4])
    ta.add_one_pattern_node([6,3])
    ta.add_one_pattern_node([3,6])
    ta.start_training()
    print(ta.forward(torch.tensor([[6,3]],dtype=torch.int)))

