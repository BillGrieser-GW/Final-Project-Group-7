"""
Class to read Street View House Numbers dataset

bgrieser
"""
import h5py
from svhnpickletypes import SvhnDigit 
        
#
# Based on a stack overflow answer:
# https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn
#
class SvhnReader():
    
    def __init__(self, filename):
        self.filename = filename
        self.h5file = h5py.File(self.filename, 'r')
        self.names = self.h5file.get('digitStruct/name')
        self.boxes = self.h5file['/digitStruct/bbox']
        self.length = len(self.names)
    
    def get_name(self, index):
        """
        Get filename for a given index
        """
        return ''.join([chr(v[0]) for v in self.h5file[self.names[index][0]].value])
    
    def get_bbox(self, index):
        """
        Get bounding boxes and labels for a given index
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
                vals.append(int(obj[0][0]))
            else:
                for k in range(obj.shape[0]):
                    vals.append(int(self.h5file[obj[k][0]][0][0]))
            box_data[name] = vals
    
        self.h5file[self.boxes[index][0]].visititems(collect_attrs)
        return box_data
    
    def get_digits_for_image(self, index):
        """
        Return a list of flat structures, one per digit identified in the image
        """
        entries = []
        file_name = self.get_name(index)
        boxes = self.get_bbox(index)
        
        for idx in range (len(boxes['label'])):
            
            tidy_entry = SvhnDigit(file_name=file_name, seq_in_file=idx+1,
                                  label=boxes['label'][idx], left=boxes['left'][idx],
                                  top=boxes['top'][idx], width=boxes['width'][idx],
                                  height=boxes['height'][idx])
            entries.append(tidy_entry)
            
        return entries
            
    def get_all_digits(self):
        """
        Get a list of tidy entries for every subimage in all the files
        """
        master_list= []
        
        for idx in range(self.length):
            master_list.append(self.get_digits_for_image(idx))
            
        return [x for el in master_list for x in el]
    
    def close(self):
        self.h5file.close()
        
        
            
            
        
    

