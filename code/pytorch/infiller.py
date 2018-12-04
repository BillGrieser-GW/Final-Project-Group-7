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

from svhnpickletypes import SvhnDigit
from svhndatasets import SvhnDigitsDataset
from PIL import Image

# Identify the model to use
STORED_MODEL = os.path.join("results", "bill_net1_1118_195850.pkl")

IMAGE_SIZE = (46,46)
CHANNELS = 1
INPUT_SIZE = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1]) 
FORCE_CPU = False
GRAD_PERCENTILE = 10

CLASSES = [str(x) for x in range(10)]
num_classes = len(CLASSES)

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
# Define a transformation that converts an image to a tensor and normalizes
# each channel
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(IMAGE_SIZE),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)])

normalizer = nn.Softmax(dim=1)

to_PIL = transforms.Compose([transforms.ToPILImage()])

to_gray_tensor = transforms.Compose([transforms.Grayscale(),
                                     transforms.ToTensor()])

DATA_DIR = os.path.join("..", "..", "data")

# Open the pickle of test parent images
print("Reading test data pickles")
with open(os.path.join(DATA_DIR, "test_parent_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading test data.")

# Instantiate a model
net = ConvNet(num_classes, CHANNELS, IMAGE_SIZE).to(device=run_device)
print(net)
total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

# Load weights
net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
print("Loading model from: ", STORED_MODEL)

total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

#%%

def imshowax(ax, img, cmap='Greys_r'):
    #img = img / 2 + 0.5
    if type(img) == torch.Tensor:
        npimg = img.numpy()
    else:
        npimg = img
        
    #ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg, cmap=cmap, interpolation='none')
    ax.tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
                   labelbottom=False, labelleft=False)
while True:
    parent_idx = input("Enter an index from 0 to {0} from the test data (q to quit): ".format(len(test_data)))
    
    try:
        if parent_idx.lower() == 'q':
            break
        
        parent_idx = int(parent_idx)
        
    except:
        print("Bad input -- assuimg 0")
        image_idx = 0

    parent_image = test_data[parent_idx].parent_image

    for digit in test_data[parent_idx].digit_data:
        
        # Extract this digit
        digit_image = parent_image.crop(digit.get_crop_box())
        digit_label = digit.label
        
        # Predict this image
        imagev = Variable(transform(digit_image)).to(device=run_device).view(1,1,IMAGE_SIZE[0], IMAGE_SIZE[1])
        imagev.requires_grad_(True)
        
        # Make the grad for the last layer by setting the predicted class
        # to 1 and all the others 0 in a vector of grads
        outputs = net(imagev)
        _, predicted = torch.max(outputs.data, 1)
        softmaxed = normalizer(outputs)[0]
        pclass = int(predicted.cpu())
        print ("Predicted class:", pclass)
        # =============================================================================
        #  GRADS
        # =============================================================================
#        last_grad = [0] * len(CLASSES)
#        last_grad[pclass] = 1
#        outputs.backward(torch.Tensor(last_grad).view(1,-1), retain_graph=True)
#       
#        # Get the grads with respect to input for the predicted output
#        thesegrads = imagev.grad[0,0].clone()
#        
#        # normalize the grads to a range 0 to 1
#        thesegrads = (thesegrads - thesegrads.min()) / (thesegrads.max() - thesegrads.min())
        
        other_grads = []
        wavg = torch.zeros((net.image_size[0],net.image_size[1]), device=run_device)
        weightsum = 0
       
        for idx in range(len(CLASSES)):
            
            if imagev.grad is not None:
                imagev.grad.data.zero_()
            
            # Select the output we want grads for
            last_grad = [0] * len(CLASSES)
            last_grad[idx] = 1
            
            outputs = net(imagev).to(device=run_device)
            
            outputs.backward(torch.Tensor(last_grad).to(device=run_device).view(1,-1), retain_graph=True)
            classgrads = imagev.grad[0,0].clone()
            
            # normalize
            classgrads = (classgrads - classgrads.min()) / (classgrads.max() - classgrads.min())
            
            if idx == pclass:
                pred_grads = classgrads.clone()
            else:
                other_grads.append(classgrads)
                wavg += classgrads
                weightsum += 1
                #wavg += (1-softmaxed[idx]) * classgrads
                #weightsum += 1-float(softmaxed[idx])
                
        wavg = wavg / weightsum
        thesegrads = abs(pred_grads - torch.tensor(wavg).type(torch.float)).cpu()
                
        # =============================================================================
        # IDENTIFY KEY PIXELS        
        # =============================================================================
        # Select the pixels with grads less than a percentile
        threshold_lo = np.percentile(thesegrads.detach().numpy().flatten(), 0)
        threshold_hi = np.percentile(thesegrads.detach().numpy().flatten(), 8)
        quiet_pixels = 128 * ((thesegrads > threshold_lo) & (thesegrads < threshold_hi))
        
        # Get back to a PIL Image matching the original size
        quiet_image = to_PIL(quiet_pixels.view(CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1])).resize((digit.width, digit.height)) 
        
        # Get a the image being infilled as a tensor
        gray_tensor = to_gray_tensor(digit_image).to(device=run_device)
        
        # We will visualize the pixels being used to find the background
        key_pixel_image = quiet_image.copy()
        
        # Selected pixels are potential traning data for the RBF net
        candidates = []
        for x in range(quiet_image.width):
            for y in range(quiet_image.height):
                if quiet_image.getpixel((x,y)) > 0:
                    candidates.append((x, y, gray_tensor[0, y, x]))
                    
        # Get a list of values used by the candidates
        values = np.array([c[2].item() for c in candidates])
        val_median = np.percentile(values, 50)
        val_mean = values.mean()
        
        if val_median < val_mean:
            # We think the region we are mtching is dark, cut off lighter candidates
            key_pixels = [c for c in candidates if c[2] < (val_median + (values.std() * 0.5))]
            print("Detecting dark target.")
        else:
            # We think the region we are matching is light; cut off darker candidates
            key_pixels = [c for c in candidates if c[2] > (val_median - (values.std() * 0.5))]
            print("Detecting light target.")
             
        # Add corners to the key pixels
        key_pixels.append((0, 0, gray_tensor[0, 0, 0]))
        key_pixels.append((digit_image.width-1, 0, gray_tensor[0, 0, digit_image.width-1]))
        key_pixels.append((digit_image.width-1, digit_image.height-1, gray_tensor[0, digit_image.height-1, digit_image.width-1]))
        key_pixels.append((0, digit_image.height-1, gray_tensor[0, digit_image.height-1, 0]))
        
        # =============================================================================
        # Train fill network      
        # =============================================================================
        fnet = fillnet.FillNet(sigma=(1.1), image_width=digit_image.width, image_height=digit_image.height, 
                               device=run_device).to(device=run_device)
        
        # Load training data
        for c in key_pixels:
            fnet.add_one_pattern_node((c[0], c[1]))
            key_pixel_image.putpixel((c[0], c[1]), 255)
        
        train_set = Variable(torch.tensor([[c[0], c[1]] for c in key_pixels], dtype=torch.int, device=run_device))
        labels = Variable(torch.tensor([c[2] for c in key_pixels], dtype=torch.float, device=run_device))
        
        fnet.start_training()
        learning_rate = 0.5
        
        # Train the model
        criterion = nn.MSELoss()
        #optimizer = torch.optim.SGD(fnet.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adagrad(fnet.parameters(), lr=learning_rate)
        
        FILL_TRAIN_EPOCHS = 2000
        BATCH_SIZE = 1000
        start_infill_train_time = time.time()
#        for epoch in range(FILL_TRAIN_EPOCHS):
#            
#            optimizer.zero_grad()
#            outputs = fnet(train_set)
#        
#            loss = criterion(outputs, labels)
#            loss.backward()
#            optimizer.step()
#            
#            #if (epoch+1) %100 == 0:
#            #    print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
#                
#            if loss.item() < 0.0001 or epoch+1 == FILL_TRAIN_EPOCHS:    
#                print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
#                break
            
        for epoch in range(FILL_TRAIN_EPOCHS):
            
            for idx in range(0, len(train_set), BATCH_SIZE):
                optimizer.zero_grad()
                toutputs = fnet(train_set[idx:idx+BATCH_SIZE])
                loss = criterion(toutputs, labels[idx:idx+BATCH_SIZE])
                loss.backward()
                optimizer.step()
            
            #if (epoch+1) %100 == 0:
            #    print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
                
            outputs = fnet(train_set)
            loss = criterion(outputs, labels)
            if loss.item() < 0.0001 or epoch+1 == FILL_TRAIN_EPOCHS:    
                print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))
                break
        
        end_infill_train_time = time.time()
        print("Found weights in {0} seconds:\n".format(int(end_infill_train_time - start_infill_train_time)),
              fnet.W2)
        
        filled_image = digit_image.copy()
        pmap = filled_image.load()
        
        # Predict the entire image using the fill net
        pixels = fnet.forward(torch.Tensor(fnet.coords).type(torch.int).to(device=run_device)).cpu().detach().numpy()
                  
        for idx, xy in enumerate(fnet.coords):
            grayp = int(255 * pixels[idx])
            pmap[xy[0], xy[1]] = (grayp, grayp, grayp)
            
        # Display
        f, ax = plt.subplots(1, 5, figsize=(11, 4.5))
        f.suptitle("Actual: {0} Predicted: {1} Parent: {2}".
                   format(CLASSES[digit_label], CLASSES[pclass], parent_idx))
        
        # ax[0].imshow(to_PIL(gray_tensor.cpu()), cmap="Greys_r")
        imshowax(ax[0], to_PIL(gray_tensor.cpu()))
        ax[0].set_xlabel("Grayscale Image for digit")  
        
        #ax[1].imshow(key_pixel_image, cmap="Blues")
        imshowax(ax[1], key_pixel_image, cmap="Blues")
        ax[1].set_xlabel("Key Pixels")  
        
        imshowax(ax[2], filled_image)
        ax[2].set_xlabel("Filled Image for digit")  
        
        ax[3].hist(([c[2].item() for c in candidates],[c[2].item() for c in key_pixels]) )
        ax[3].set_xlabel("Candidate value histogram") 
        
        ax[4].hist(imagev.cpu().detach().numpy().flatten())
        ax[4].set_xlabel("Historgram of whole image")  
        f.show()
        plt.show()