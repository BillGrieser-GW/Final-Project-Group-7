# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 11:13:32 2018

@author: billg_000
"""

import sys

# Allow imports from parent dir
sys.path.insert(0,"..")

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.ma as ma

import torch.nn as nn
from torch.autograd import Variable
from bill_nets import ConvNet

from svhnpickletypes import SvhnDigit
from svhndatasets import SvhnDigitsDataset
from PIL import Image

import pickle

import fillnet

# Identify the model to use
STORED_MODEL = os.path.join("results", "bill_net1_1118_195850.pkl")

IMAGE_SIZE = (46,46)
CHANNELS = 1
INPUT_SIZE = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1]) 
FORCE_CPU = True
GRAD_THRESHOLD = 0.30

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
fill_to_Tensor = transforms.Compose([transforms.Grayscale(),
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

def imshowax(ax, img):
    #img = img / 2 + 0.5
    if type(img) == torch.Tensor:
        npimg = img.numpy()
    else:
        npimg = img
        
    #ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg, cmap='Greys_r')
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
        last_grad = [0] * len(CLASSES)
        outputs = net(imagev)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        softmaxed = normalizer(outputs)[0]
        predicted = predicted.cpu()
        pclass = int(predicted)
        
        last_grad[pclass] = 1
        outputs.backward(torch.Tensor(last_grad).view(1,-1), retain_graph=True)
        
        # Get the grads with respect to input for the predicted output
        thesegrads = imagev.grad[0,0].clone()
        
        print ("Predicted class:", pclass)
        
        # normalize the grads to a range 0 to 1
        thesegrads = (thesegrads - thesegrads.min()) / (thesegrads.max() - thesegrads.min())
        
        # Select the pixels with grads less than a threshold
        selected_pixels = 255 * (thesegrads < GRAD_THRESHOLD)
        #selected_pixels = imagev
        
        # Get back to a PIL Image
        pimg = to_PIL(selected_pixels.view(CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1])).resize((digit.width, digit.height)) 
        
        # Get a the image being infilled as a tensor
        infill_article = fill_to_Tensor(digit_image)
        
        # Selected pixels are potential traning data for the RBF net
        candidates = []
        for x in range(pimg.width):
            for y in range(pimg.height):
                if pimg.getpixel((x,y)) > 0:
                    candidates.append((infill_article[0, y, x], x, y))
                    
        fnet = fillnet.FillNet().to(device=run_device)
        
        # Load training data
        for c in candidates:
            fnet.add_one_training_sample((c[1], c[2]), c[0])
        
        train_set = Variable(torch.tensor([[c[1], c[2]] for c in candidates], dtype=torch.int))
        labels = Variable(torch.tensor([c[0] for c in candidates], dtype=torch.float))
        
        fnet.train()
        learning_rate = 0.5
        
        # Train the model
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD([fnet.W2], lr=learning_rate)
        
        for epoch in range(50):
            
            optimizer.zero_grad()
            outputs = fnet(train_set)
        
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print("Epoch: {0} Loss: {1}".format(epoch+1, loss.item()))