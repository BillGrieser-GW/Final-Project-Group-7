"""
Train a fillnet based on a set of training pixels
"""
import time
import torch
from torch.autograd import Variable
import torch.nn as nn

LEARNING_RATE = 0.1
FILL_CHANNELS = 3

class FillNetTrainer():
    
    def __init__(self, fnet, grid_spacing=3, max_epochs=2000, batch_size=1000):
        self.fnet = fnet
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.grid_spacing = 3
        self.training_pixels = [] # 3-tuple of x, y, value
        
        if self.grid_spacing > 0:
            self.grid_coords= []
            # Generate the ideal grid coordinates for this image
            for x in range(0, fnet.image_width, grid_spacing):
                for y in range(0, fnet.image_height, grid_spacing):
                    self.grid_coords.append((x,y))
        
    def train(self, new_training_pixels, training_pixel_image=None):
        """
        Train the input fillnet with the input pixels and their values. 
        """
        self.training_pixels += new_training_pixels
        self.fnet.reset()
        
        # Load training data into the network object
        for c in self.training_pixels:
            self.fnet.add_one_pattern_node((c[0], c[1]))
            if training_pixel_image is not None:
                training_pixel_image.putpixel((c[0], c[1]), 255)
        
        train_set = Variable(torch.tensor([[c[0], c[1]] for c in self.training_pixels], dtype=torch.int, device=self.fnet.device))
        labels = Variable(torch.cat([c[2] for c in self.training_pixels]).view(-1,3)).to(device=self.fnet.device)
        
        # With the data loaded, prepare the network for training
        self.fnet.start_training()
        
        # Set training criterion
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adagrad(self.fnet.parameters(), lr=LEARNING_RATE)
        
        # Train the model
        start_infill_train_time = time.time()
    
        self.fnet.sigmaSq.requires_grad = False
        self.fnet.W2.requires_grad = True
        
        for epoch in range(self.max_epochs):
           
            for idx in range(0, len(train_set), self.batch_size):
                self.fnet.zero_grad()
                toutputs = self.fnet(train_set[idx:idx+self.batch_size])
                loss = self.criterion(toutputs, labels[idx:idx+self.batch_size].view(-1, FILL_CHANNELS))
                loss.backward()
                self.optimizer.step()
            
            #if (epoch+1) % 100 == 0:
            #    print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
                
            outputs = self.fnet(train_set)
            loss = self.criterion(outputs, labels.view(-1, FILL_CHANNELS))
            
            # Early stop if loss near 0
            if  loss.item() < 0.00005:    
                break
        
        end_infill_train_time = time.time()
        
        print("Found weights in {0} seconds in {1} epochs with loss: {2}, pattern layer size: {3}.\n".format(
                int(end_infill_train_time - start_infill_train_time),
                epoch+1, loss.item(), len(self.fnet.pattern_layer)))
        
        return self.fnet
    
    def augment_and_retrain(self):
        
        X = torch.tensor([0,0], dtype=torch.int, device='cpu')
        new_pixels = []
        
        # Get a list of the grid co-ords that are not already in the pattern layer
        pattern_coords = [tuple(x) for x in self.fnet.pattern_layer.cpu().numpy()]
        
        candidates = [x for x in self.grid_coords if x not in pattern_coords]
        
        if len(candidates) > 0:
            
            distances = []
            
            # Get the distance for each candidate to the nearest pattern layer node
            for c in candidates:
                X[0], X[1] = c
                d = (self.fnet.pattern_layer.cpu() - X).pow(2).sum(dim=1).type(torch.float).min()
                distances.append((c, d.item()))
                
            # Find min distance in the list
            min_d = min(distances, key=lambda x: x[1])[1]
            
            # Find all the coords in the grid candidates with that distance
            new_pattern_nodes = [x[0] for x in distances if x[1] == min_d]
            
            # Predict the new pattern nodes
            new_labels = self.fnet(torch.Tensor(new_pattern_nodes).type(torch.int))
            
            # Incorporate these back into the pattern layer
            new_pixels = zip(new_pattern_nodes, new_labels)
            
            # Convert the nodes into scalars instead of tuple
            new_pixels = [(x[0][0], x[0][1], x[1]) for x in new_pixels]
            
            self.train(new_pixels)
                    
        return new_pixels
        