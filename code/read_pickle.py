# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:36:33 2018

@author: billg_000
"""

import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import svhnreader

# Read a pickle file
print("Reading pickle")
with open(os.path.join("..", "data", "train_data.pkl"), 'rb') as f:
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
#tests = [40, 12002]

for t in tests:
    
    mi = train_data[t][1]
    f, ax = plt.subplots(1, len(mi)+1, figsize=(10,4))
    
    parent =  train_data[t][0]
    imshow(ax[0], parent, mi, title=train_data[t][1][0].file_name) 
    
    for d in mi:
        imshow(ax[d.seq_in_file], parent.crop(d.crop_box()), title="Label = {0}".format(d.label)) 
        
    plt.show()