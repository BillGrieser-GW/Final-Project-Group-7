# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:33:28 2018

@author: billg_000
"""

import os.path
from PIL import Image
from PIL import ImageOps

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
from torch.autograd import Variable
import datetime
from svhnpickletypes import SvhnDigit

import svhnreader

FORCE_CPU = False

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    

DATA_DIR = r"D:\Users\billg_000\GradSchool\OneDrive - gwmail.gwu.edu\DATS 6203 ML II\Final Project\SVHN\train"
METADATA_FILE = "digitStruct.mat"

data_file = os.path.join(DATA_DIR, METADATA_FILE)

def imshow(im_frame, metadata=[]):
    f, ax = plt.subplots(figsize=(6,6))
    ax.imshow(im_frame)
    
    for md in metadata:
        rect = patches.Rectangle((md.left,md.top),md.width,md.height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
    plt.show()
    
# Create a reader object
sr = svhnreader.SvhnReader(data_file)

# Prepare the
#%%

tests = [12002, 0, 12, 18, 22660, 18, 29]


for t in tests:
    mi = sr.get_digits_for_image(t)
    
    image_file = os.path.join(DATA_DIR, mi[0].file_name)
    
    im_frame = Image.open(image_file)
    imshow(im_frame, mi) 
    print ("Labels:", [x.label for x in mi])
    
    for d in mi:
        imshow(im_frame.crop(d.get_crop_box())) 
#%%
pickle_me = []
for t in range(sr.length):
    digit_data = sr.get_digits_for_image(t)
    
    image_file = os.path.join(DATA_DIR, digit_data[0].file_name)
    
    parent = Image.open(image_file)
    parent.load()
    pickle_me.append((mi[0].file_name, parent, digit_data))
    
    if ((t+1) % 200) == 0:
        print("Done with {0} entries".format(t+1))
    
print("Writing pickle")
with open(os.path.join("..", "data", "train_data.pkl"), 'wb') as f:
    pickle.dump(pickle_me, f)

print("Done writing pickle")
