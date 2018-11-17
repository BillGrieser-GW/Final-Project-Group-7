# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 08:51:15 2018

@author: billg_000
"""

import torch

# Make a dataset wrapper for the images
class SvhnDigitsDataset(torch.utils.data.Dataset):
    
    def __init__(self, digits_data, transform=lambda x:x):
        """
        digits_data: a SvhnDigitPickle 
        """
        super(torch.utils.data.Dataset, self).__init__()
        
        self.digits_data = digits_data
        self.transform = transform
       
    def __len__(self):
        return len(self.digits_data)
        
    def __getitem__(self, index):
        return self.transform(self.digits_data[index].digit_image), \
           self.digits_data[index].data.label
    
    
    