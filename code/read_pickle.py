# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:36:33 2018

@author: billg_000
"""



import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from svhnpickletypes import SvhnDigit
import pickle
   
# =============================================================================
# Parent files == parent image and metadate for each digit in image
# =============================================================================
# Read a pickle file
print("Reading pickle")
with open(os.path.join("..", "data", "train_parent_data.pkl"), 'rb') as f:
    train_data = pickle.load(f)

print("Done Reading pickle")

def imshow(ax, im_frame, metadata=[], title=''):
    
    ax.imshow(im_frame)
    
    for md in metadata:
        rect = patches.Rectangle((md.left,md.top),md.width,md.height,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
    if title != '':
        ax.set_title(title)
        
#%% 
tests = [40, 12002, 0, 12, 18, 22660, 19, 29]
#tests = [49, 44, 0, 1, 2, 3, 18, 19, 40]

for t in tests:
    
    mi = train_data[t]
    
    f, ax = plt.subplots(1, len(mi.digit_data) + 1, figsize=(10,4))
    
    parent =  train_data[t].parent_image
    imshow(ax[0], parent, mi.digit_data, title=mi.file_name) 
    
    for d in mi.digit_data:
        imshow(ax[d.seq_in_file], parent.crop(d.get_crop_box()), title="Label = {0}".format(d.label)) 
        
    plt.show()
#%%    
# =============================================================================
# Parent files == parent image and metadate for each digit in image
# =============================================================================
# Read a pickle file
with open(os.path.join("..", "data", "train_digit_data.pkl"), 'rb') as f:
    train_data = pickle.load(f)
    
for t in tests:
    
    dd = train_data[t]
    
    f, ax = plt.subplots(figsize=(10,4))
    imshow(ax, dd.digit_image, title="Label = {0}".format(dd.data.label)) 
    
    plt.show()