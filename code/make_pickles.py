"""
Read the default Street View House Numbers data and build pickled
versions of the data in a format easier to work with.

bgrieser
"""
import os
import os.path
from PIL import Image
import svhnreader
from svhnpickletypes import SvhnParentPickle, SvhnDigitPickle
import pickle

# Assume the data is kept in a folder that is a sibling to the folder where the code is kept
DATA_PARENT = os.path.realpath(os.path.join("..", "data"))
METADATA_FILE = "digitStruct.mat"

VERSIONS_TO_MAKE = ("train", "test")

for version in VERSIONS_TO_MAKE:
    data_dir = os.path.join(DATA_PARENT, version)
    print("Processing data for version", version)
    data_file = os.path.join(data_dir, METADATA_FILE)
    print("   from dir:", data_dir)
    
    # Create a reader object
    sr = svhnreader.SvhnReader(data_file)
    
    print("{0:d} entries found in file".format(sr.length))
    pickle_me_parents = []
    pickle_me_digits = []
    
    for t in range(sr.length):
        digit_data = sr.get_digits_for_image(t)
        
        image_file = os.path.join(data_dir, digit_data[0].file_name)
        
        parent_image = Image.open(image_file)
        parent_image.load()
        
        # Make an object for this parent file
        p = SvhnParentPickle(digit_data[0].file_name, parent_image, digit_data) 
        pickle_me_parents.append(p)
                     
        # Make digits object for each of the digits
        for digit in digit_data:
            d = SvhnDigitPickle(parent_image.crop(digit.get_crop_box()), digit)
            pickle_me_digits.append(d)
        
        if ((t+1) % 200) == 0:
            print("Done with {0} entries".format(t+1))
    
    sr.close()
    
    print("Writing parents pickle")
    with open(os.path.join(DATA_PARENT,"{0}_parent_data.pkl".format(version)), 'wb') as f:
        pickle.dump(pickle_me_parents, f)
    print("Writing digits pickle")
    with open(os.path.join(DATA_PARENT,"{0}_digit_data.pkl".format(version)), 'wb') as f:
        pickle.dump(pickle_me_digits, f)
        
    print("Done writing pickles for {0}.\n".format(version))
   
    
   
    
