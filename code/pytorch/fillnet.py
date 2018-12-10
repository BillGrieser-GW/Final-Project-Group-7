"""
Implementation of a GRNN to do image infill using techniques based on
paper:
    
Application of GRNN neural network in non-texture image inpainting and restoration
Vahid K. Aliloua, Farzin Yaghmaeeb

with modificications
"""
import torch
import torch.nn as nn
import numpy as np
import PIL

class FillNet(nn.Module):
    
    def __init__(self, image_width, image_height, channels=1, device='cpu', \
                 sigma=np.e, adapt_sigma=True):
        """
        Initialize a network which will be filled out as training
        samples are added. Sigma should be 1 or greater.
        """
        super(FillNet, self).__init__()
        self.device=device
        self.sigma = sigma
        self.image_width = image_width
        self.image_height = image_height
        self.channels = channels
        self.adapt_sigma = adapt_sigma
        self.reset()
        
    def reset(self):
        self.pattern_layer = None
        self.pattern_node_set = set()
        self.pattern_node_list = []
        self.W2 = None
        self.sigmaSq = None
        self.Dsquares = dict()
        
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
        self.pattern_layer = torch.tensor(np.vstack(self.pattern_node_list), dtype=torch.int).to(self.device)
        
        # Initialize the weights for each channel
        #self.W2 = nn.Parameter(torch.rand((self.channels, len(self.pattern_node_list)), requires_grad=True, 
        #                                  device=self.device))
       
        self.W2 = nn.Parameter(torch.full((self.channels, len(self.pattern_node_list)), 1, requires_grad=True, 
                             device=self.device))

        # Generate the Dsquared from all points to the pattern layer
        self.coords = [(x, y) for x in range(self.image_width) for y in range(self.image_height)]
        
        X = torch.tensor([0,0], dtype=torch.int, device='cpu')
        
        for idx, xy in enumerate(self.coords):
            X[0], X[1] = xy    
            self.Dsquares[xy] = (self.pattern_layer - X.to(self.device)).pow(2).sum(dim=1).type(torch.float).to(device=self.device)
        
        self.sigmaSq = torch.full((len(self.pattern_node_list),), self.sigma**2, 
                                               requires_grad=False, dtype=torch.float).to(device=self.device)
        if self.adapt_sigma:
            for idx in range(len(self.pattern_layer)):
                self.my_squares = self.Dsquares[tuple(self.pattern_layer[idx].numpy())]
                min_not_me = torch.cat((self.my_squares[0:idx], self.my_squares[idx+1:])).min()
                self.sigmaSq[idx] = 1.0 + (np.log(min_not_me) * 1.5) 
        
    def rbf(self, W2, Dsquared):
        return W2 * (torch.exp(-1.0 * Dsquared / (2.0 * self.sigmaSq)))
        
    def forward(self, X):
        self.X = X
        out = torch.zeros(len(X), dtype=torch.float, device=self.device)
        
        self.Dsquared = torch.zeros((len(X), len(self.pattern_layer)), dtype=torch.float).to(device=self.device)
        
        for idx in range(len(X)):
            
           # Get sqaured distances to all pattern layer points from this X by
           # getting them from the precomputed values
           self.Dsquared[idx] = self.Dsquares[tuple(X[idx].cpu().numpy())]
           
        self.channel_out = []
        
        for cidx in range(self.channels):
            self.channel_out.append(self.rbf(self.W2[cidx], self.Dsquared).sum(dim=1) / self.rbf(1, self.Dsquared).sum(dim=1))
            
        # Repackage the channels so that all channels for a pixel are on the same row
        out = torch.cat(self.channel_out, dim=0).view(self.channels, -1).transpose(0,1)
        
        return out
    
    def generate_image(self):
        """
        Use the trained network to generate a PIL image with the given image width and height
        """
        out = PIL.Image.new("RGB", (self.image_width, self.image_height))
        
        pmap = out.load()
        
        # Predict the entire image using the fill net
        pixels = self.forward(torch.Tensor(self.coords).type(torch.int).to(device=self.device)).cpu().detach().numpy()
                  
        for idx, xy in enumerate(self.coords):
            try:
                pmap[xy[0], xy[1]] = tuple((255 * pixels[idx]).astype(int))
            except:
                pmap[xy[0], xy[1]] = (0,0,0)
                print("Error for pixel:", xy)
            
        return out
        
#    def parameters(self):
#        """
#        return a list of parameters
#        """
#        return [self.W2, self.sigmaSq]
    
if __name__ == "__main__":
    ta = FillNet(7,5)
    
    ta.add_one_pattern_node([2,4])
    ta.add_one_pattern_node([6,3])
    ta.add_one_pattern_node([3,2])
    ta.start_training()
    print(ta.forward(torch.tensor([[6,3]],dtype=torch.int)))

