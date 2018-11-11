# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:25:39 2018

@author: billg_000
"""
import os.path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import datetime
import pickle

FORCE_CPU = False

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
DATA_DIR = r"D:\Users\billg_000\GradSchool\OneDrive - gwmail.gwu.edu\DATS 6203 ML II\Final Project\dataset5"

TRAIN_FOLDERS = ['A', 'B', 'D', 'E']
TEST_FOLDER = ['C']

TEST_FILE = os.path.join(DATA_DIR, "B", "e", "color_4_0003.png")
classes = ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y')


im_frame = Image.open(TEST_FILE)

def imshow(im_frame):
    f, ax = plt.subplots(figsize=(6,6))
    ax.imshow(im_frame)
    plt.show()

transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                
# Make a dataset wrapper for the images
class images_dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super(torch.utils.data.Dataset, self).__init__()
        
        self.images = []
        self.labels = []
        self.widths = []
        self.heights = []
        self.sources = []
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.sources[index]
    
    def add_item(self, image, label, source):
        self.images.append(image)
        self.labels.append(label)
        self.sources.append(source)

def load_dataset(dataset, sources=["A"], skip_in_class=0, limit_in_class=10000):
    """Load the daset with the images from the sources specified by the 
    input parameter"""
    for sidx, source in enumerate(sources):
        counters = np.zeros((len(sources), len(classes)))
        
        for root, dirs, files in os.walk(os.path.join(DATA_DIR, source), topdown=True):
    
            # Process all the files
            for fname in files:
                if fname.startswith("color_"):
                    
                     # Get the label
                    label = int(fname.split('_')[1])
                    
                    # Account for the missing j
                    if label > 9:
                        label = label - 1
        
                    # Do we have enough of this label from this source?
                    counters[sidx, label] += 1
                    
                    if counters[sidx, label] > skip_in_class and \
                        counters[sidx, label] <= (limit_in_class + skip_in_class):
                            
                        # Process this file
                        im_frame = Image.open(os.path.join(root, fname))
                        image = (transform(im_frame))
                        train_dataset.add_item(image, label, source)
        
train_dataset = images_dataset()


load_dataset(train_dataset, sources=['A'])

#%%        
print("Writing pickle")
with open("train_data.pkl", 'wb') as f:
    pickle.dump(train_dataset, f)

#%%
    
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    
for i, data in enumerate(train_loader):
    print (i, len(data), len(data[0]))
    if i > 3:
        break

    
print("Done")

