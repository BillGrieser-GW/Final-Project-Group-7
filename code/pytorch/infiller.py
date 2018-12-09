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
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable

import predictor_nets
import fillnet
from fillnet_trainer import FillNetTrainer
import key_pixels

from plot_helpers import imshowax
# Identify the model to use
#STORED_MODEL = os.path.join("results", "basis_runs", "train_predictor_1208_230417.pkl")
STORED_MODEL = os.path.join("results", "basis_runs", "Ctrain_predictor_1209_182340.pkl")
FEATURE_MAPS = 48

DATA_DIR = os.path.join("..", "..", "data")
IMAGE_SIZE = (40,40)
PREDICT_CHANNELS = 1
GRAD_PERCENTILE = 10
FILL_CHANNELS = 3
GRID_SPACING = 3

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
net = predictor_nets.ConvNet48_Dropout3(num_classes, PREDICT_CHANNELS, IMAGE_SIZE).to(device=run_device)
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
    parent_idx = input("Enter an index from 0 to {0} from the test data (q to quit): ".format(len(test_data)-1))
    
    try:
        if parent_idx.lower() == 'q':
            break
        parent_idx = int(parent_idx)
        
    except:
        print("Bad input -- assuimg 0")
        parent_idx = 0

    if parent_idx >= len(test_data) or parent_idx < 0:
        print("Bad input -- assuimg 0")
        parent_idx = 0
        
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
        #training_pixels, quiet_image, candidates = kpf.get_using_grad_near_average(digit_image, pclass, imagev)
        #training_pixels, quiet_image, candidates = kpf.get_using_grad_value(digit_image, pclass, imagev)
        #training_pixels, quiet_image, candidates = kpf.get_using_grid(digit_image, pclass, imagev)
        training_pixels, quiet_image, candidates = kpf.get_using_fmaps_std(digit_image, 
                                          pclass, imagev, grid_spacing=GRID_SPACING)
        
        # We will visualize the pixels being used to find the background
        training_pixel_image = quiet_image.copy()
        
        # =============================================================================
        # Make a train network to use to fill the image    
        # =============================================================================
        fnet = fillnet.FillNet(sigma=(1.8), adapt_sigma=False, image_width=digit_image.width,
                               image_height=digit_image.height, channels=FILL_CHANNELS, 
                               device=run_device).to(device=run_device)

        trainer = FillNetTrainer(fnet)
        
        trainer.train(training_pixels, training_pixel_image, )
        
        # Synthesize more training data from the initial batch in order to 
        # get a smoother image
        idx = 0
        while idx < 30:
            idx += 1
            new_pixels = trainer.augment_and_retrain()
            if len(new_pixels) == 0:
                break
            
        # Use the trained network to generate an image
        filled_image = fnet.generate_image()
        
        # =============================================================================
        # Display
        # =============================================================================
        
        f, all_ax = plt.subplots(1, 3, figsize=(10, 7))
        f.suptitle("Actual: {0} Predicted: {1} Parent: {2}".
                   format(CLASSES[digit_label], CLASSES[pclass], parent_idx))
        
        ax = all_ax
        imshowax(ax[0], digit_image)
        #ax[0].imshow(digit_image)
        ax[0].set_xlabel("Image for digit") 
        
        #ax[1].imshow(key_pixel_image, cmap="Blues")
        imshowax(ax[1], training_pixel_image, cmap="Blues")
        ax[1].set_xlabel("Key Pixels")  
        
        imshowax(ax[2], filled_image)
        ax[2].set_xlabel("Filled Image for digit")  
        
#        ax[3].hist(([c[2].mean().item() for c in candidates], [c[2].mean().item() for c in training_pixels]) )
#        ax[3].set_xlabel("Training value histogram") 
#        
#        ax[4].hist(imagev.cpu().detach().numpy().flatten())
#        ax[4].set_xlabel("Historgram of whole image")  
        
        filled_parent.paste(filled_image, digit.get_crop_box())
        #print("SigmaSq:", fnet.sigmaSq)
        
        f, all_ax = plt.subplots(int(FEATURE_MAPS/8), 8, figsize=(12, 6))
        f.suptitle("Feature maps for Actual: {0} Predicted: {1} Parent: {2}".
                   format(CLASSES[digit_label], CLASSES[pclass], parent_idx))
                   
        fmaps = net.layer1[0].forward(imagev).detach()
        
        for rg in range(int(FEATURE_MAPS/8)):
            for cg in range(8):
                imshowax(all_ax[rg, cg], fmaps[0, rg*8 + cg])
                all_ax[rg, cg].set_xlabel("{0}".format(rg*8 + cg))
                
        f, all_ax = plt.subplots(1, 4, figsize=(11, 6))
        f.suptitle("Fmap mean & Std Deviation for Actual: {0} Predicted: {1} Parent: {2}".
                   format(CLASSES[digit_label], CLASSES[pclass], parent_idx))
        imshowax(all_ax[0], fmaps[0].mean(dim=0))
        all_ax[0].set_xlabel("Feature Map Mean")
        imshowax(all_ax[1], fmaps[0].std(dim=0))
        all_ax[1].set_xlabel("Feature Map Std Deviation")
        imshowax(all_ax[2], fmaps[0].mean(dim=0) - fmaps[0].std(dim=0))
        all_ax[2].set_xlabel("Feature mean - std")
        all_ax[3].hist((fmaps[0].mean(dim=0) - fmaps[0].std(dim=0)).flatten())
        all_ax[3].set_xlabel("Feature mean - std hist")
        
        #imshowax(all_ax[3], fmaps[0].mean(dim=0).pow(fmaps[0].std(dim=0)))
        #all_ax[3].set_xlabel("Feature mean/std")
        
        #print(net.layer1[0].weight.grad.mean(dim=1).mean(dim=1).mean(dim=1) + net.layer1[0].bias)
        
    f, ax = plt.subplots(1, 2, figsize=(10, 3))
    f.suptitle("Parent Images before and digit replacement (Image {0})".format(parent_idx))
    imshowax(ax[0], parent_image)
    ax[0].set_xlabel("Original")  
    imshowax(ax[1], filled_parent)
    ax[1].set_xlabel("Altered")  
    plt.show()