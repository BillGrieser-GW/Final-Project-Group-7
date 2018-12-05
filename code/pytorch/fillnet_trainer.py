"""
Train a fillnet based on a set of training pixels
"""
import time
import torch
from torch.autograd import Variable
import torch.nn as nn

LEARNING_RATE = 0.1
FILL_CHANNELS = 3

def train(fnet, training_pixels, training_pixel_image=None, max_epochs=2000, batch_size=1000):
    """
    Train the input fillnet with the input pixels and their values. 
    """
    
    # Load training data into the network object
    for c in training_pixels:
        fnet.add_one_pattern_node((c[0], c[1]))
        if training_pixel_image is not None:
            training_pixel_image.putpixel((c[0], c[1]), 255)
    
    train_set = Variable(torch.tensor([[c[0], c[1]] for c in training_pixels], dtype=torch.int, device=fnet.device))
    labels = Variable(torch.cat([c[2] for c in training_pixels]).view(-1,3)).to(device=fnet.device)
    
    # With the data loaded, prepare the network for training
    fnet.start_training()
    
    # Set training criterion
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(fnet.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    start_infill_train_time = time.time()

    fnet.sigmaSq.requires_grad = False
    fnet.W2.requires_grad = True
    
    for epoch in range(max_epochs):
        
        for idx in range(0, len(train_set), batch_size):
            fnet.zero_grad()
            toutputs = fnet(train_set[idx:idx+batch_size])
            loss = criterion(toutputs, labels[idx:idx+batch_size].view(-1, FILL_CHANNELS))
            loss.backward()
            optimizer.step()
        
        #if (epoch+1) % 100 == 0:
        #    print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
            
        outputs = fnet(train_set)
        loss = criterion(outputs, labels.view(-1, FILL_CHANNELS))
        
        # Early stop if 
        if loss.item() < 0.00005:    
            break
    
    end_infill_train_time = time.time()
    
    print("Found weights in {0} seconds in {1} epochs with loss: {2}.\n".format(
            int(end_infill_train_time - start_infill_train_time),
            epoch+1, loss.item()))
    
    return fnet