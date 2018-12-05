# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:13:32 2018

@author: billg_000
"""

import sys


# Allow imports from parent dir
sys.path.insert(0,"..")

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable

from bill_nets import ConvNet
import fillnet
import key_pixels

from plot_helpers import imshowax
# Identify the model to use
STORED_MODEL = os.path.join("results", "bill_net1_1118_195850.pkl")
DATA_DIR = os.path.join("..", "..", "data")

IMAGE_SIZE = (46,46)
PREDICT_CHANNELS = 1
GRAD_PERCENTILE = 10
FILL_CHANNELS = 3

CLASSES = [str(x) for x in range(10)]
num_classes = len(CLASSES)

FORCE_CPU = False
if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
normalizer = nn.Softmax(dim=1)

# Make transforms for converting to and from various formats
to_PIL = transforms.Compose([transforms.ToPILImage()])

# Open the pickle of test parent images
print("Reading test data pickles")
with open(os.path.join(DATA_DIR, "test_parent_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading test data.")

# Instantiate a model to use to predict
net = ConvNet(num_classes, PREDICT_CHANNELS, IMAGE_SIZE).to(device=run_device)
transform = net.get_transformer()

print(net)

total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

# Load weights
net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
print("Loading model from: ", STORED_MODEL)

total_net_parms = net.get_total_parms()

#%%
while True:
    parent_idx = input("Enter an index from 0 to {0} from the test data (q to quit): ".format(len(test_data)))
    
    try:
        if parent_idx.lower() == 'q':
            break
        parent_idx = int(parent_idx)
        
    except:
        print("Bad input -- assuimg 0")
        image_idx = 0

    # Process this parent image
    parent_image = test_data[parent_idx].parent_image
    filled_parent = parent_image.copy()
    
    for digit in test_data[parent_idx].digit_data:
        
        # Extract this digit
        digit_image = parent_image.crop(digit.get_crop_box())
        digit_label = digit.label
        
        # Predict this image
        imagev = Variable(transform(digit_image)).to(device=run_device).view(1,1,IMAGE_SIZE[0],IMAGE_SIZE[1])
        imagev.requires_grad_(True)
        
        outputs = net(imagev)
        _, predicted = torch.max(outputs.data, 1)
        softmaxed = normalizer(outputs)[0]
        pclass = int(predicted.cpu())
        print ("Predicted class:", pclass)
        
        # =============================================================================
        # Find pixels from the image to use as training data for the image fill   
        # =============================================================================
        kpf = key_pixels.KeyPixelFinder(net, CLASSES, device=run_device)
        training_pixels, quiet_image, candidates = kpf.get_using_grad_near_average(digit_image, pclass, imagev)
        
        # We will visualize the pixels being used to find the background
        training_pixel_image = quiet_image.copy()
        
        # =============================================================================
        # Make a train network to use to fill the image    
        # =============================================================================
        fnet = fillnet.FillNet(sigma=(np.e/2 + 0.5), image_width=digit_image.width, image_height=digit_image.height, 
                               channels=FILL_CHANNELS, device=run_device).to(device=run_device)
        
        # Load training data
        for c in training_pixels:
            fnet.add_one_pattern_node((c[0], c[1]))
            training_pixel_image.putpixel((c[0], c[1]), 255)
        
        train_set = Variable(torch.tensor([[c[0], c[1]] for c in training_pixels], dtype=torch.int, device=run_device))
        labels = Variable(torch.cat([c[2] for c in training_pixels]).view(-1,3)).to(device=run_device)
        
        fnet.start_training()
        learning_rate = 0.1
        
        # Train the model
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(fnet.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adagrad(fnet.parameters(), lr=learning_rate)
        
        FILL_TRAIN_EPOCHS = 2000
        BATCH_SIZE = 1000
        start_infill_train_time = time.time()

        fnet.sigmaSq.requires_grad = False
        fnet.W2.requires_grad = True
        
        for epoch in range(FILL_TRAIN_EPOCHS):
            
            #fnet.sigmaSq.requires_grad = (epoch % 2 == 1)
            #fnet.W2.requires_grad = (epoch % 2 == 0)
            
            for idx in range(0, len(train_set), BATCH_SIZE):
                fnet.zero_grad()
                toutputs = fnet(train_set[idx:idx+BATCH_SIZE])
                loss = criterion(toutputs, labels[idx:idx+BATCH_SIZE].view(-1, FILL_CHANNELS))
                loss.backward()
                optimizer.step()
            
            #if (epoch+1) %100 == 0:
            #    print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
                
            outputs = fnet(train_set)
            loss = criterion(outputs, labels.view(-1, FILL_CHANNELS))
            if loss.item() < 0.00005 or epoch+1 == FILL_TRAIN_EPOCHS:    
                print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
                break
        
        end_infill_train_time = time.time()
        print("Found weights in {0} seconds:\n".format(int(end_infill_train_time - start_infill_train_time)),
              fnet.W2)
        
        # Use the trained network to generate an image
        filled_image = fnet.generate_image()
        
        # Display
        f, all_ax = plt.subplots(1, 5, figsize=(10, 7))
        f.suptitle("Actual: {0} Predicted: {1} Parent: {2}".
                   format(CLASSES[digit_label], CLASSES[pclass], parent_idx))
        
        # ax[0].imshow(to_PIL(gray_tensor.cpu()), cmap="Greys_r")
        #imshowax(ax[0], to_PIL(gray_tensor.cpu()))
        #ax[0].set_xlabel("Grayscale Image for digit") 
        ax = all_ax
        imshowax(ax[0], digit_image)
        #ax[0].imshow(digit_image)
        ax[0].set_xlabel("Image for digit") 
        
        #ax[1].imshow(key_pixel_image, cmap="Blues")
        imshowax(ax[1], training_pixel_image, cmap="Blues")
        ax[1].set_xlabel("Key Pixels")  
        
        imshowax(ax[2], filled_image)
        ax[2].set_xlabel("Filled Image for digit")  
        
        ax[3].hist(([c[2].mean().item() for c in candidates], [c[2].mean().item() for c in training_pixels]) )
        ax[3].set_xlabel("Training value histogram") 
        
        ax[4].hist(imagev.cpu().detach().numpy().flatten())
        ax[4].set_xlabel("Historgram of whole image")  
        
        filled_parent.paste(filled_image, digit.get_crop_box())
        print("SigmaSq:", fnet.sigmaSq)
        
    f, ax = plt.subplots(1, 2, figsize=(10, 3))
    f.suptitle("Parent Images before and digit replacement (Image {0})".format(parent_idx))
    imshowax(ax[0], parent_image)
    ax[0].set_xlabel("Original")  
    imshowax(ax[1], filled_parent)
    ax[1].set_xlabel("Altered")  
    plt.show()