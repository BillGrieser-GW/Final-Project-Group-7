# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:33:28 2018

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

import scipy.io
import h5py

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
#%%
snf = h5py.File(data_file, 'r')

#%%
bboxes = snf.get('digitStruct/bbox')

#
# Based on a stack overflow answer:
# https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
#
def get_name(index, snf, names):
    names = snf.get('digitStruct/name')
    return ''.join([chr(v[0]) for v in snf[names[index][0]].value])

def get_box_data(index, snf):
    """
    get bounding boxes for a given index
    """
    box_data = dict()
    box_data['height'] = []
    box_data['label'] = []
    box_data['left'] = []
    box_data['top'] = []
    box_data['width'] = []

    def collect_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(snf[obj[k][0]][0][0]))
        box_data[name] = vals

    box = snf['/digitStruct/bbox'][index]
    snf[box[0]].visititems(collect_attrs)
    return box_data

