"""
Classes and command-line operations to convert the SVHN data into pickled
formats for tidier use.

bgrieser
"""

# =============================================================================
# Class definition for what gets pickled
# =============================================================================
class SvhnDigitPickle():
    """
    This class represents one entry in the a of data where each entry is
    the image and metadata for one digit.
    """
    def __init__(self, digit_image, data):
        """
        digit_image: a PIL image of the digit
        data: metadata for the digit, type SvhnDigit
        """
        self.digit_image = digit_image
        self.data = data


class SvhnParentPickle():
    """
    This class represents one entry in a list of data where each entry is
    the parent image and metedata for all the digits that it contains.
    """
    def __init__(self, file_name, parent_image, digit_data):
        """
        file_name: The original file name of the parent image
        parent_image: a PIL image of the parent image containing one or more digits
        digit_data: a list of SvhnDigit classes, one per digit in the image
        """
        self.parent_image = parent_image
        self.digit_data = digit_data
        self.file_name = file_name
        
        
class SvhnDigit():
    """
    This class defines the metadata for one digit. It provides convenience
    functions to generate the crop box to crop the digit image out of its
    parent image, and to calculate the padding required to center the digit
    image in a given image size.
    """
    def __init__(self,  file_name, seq_in_file, label, left, top, width, height):
        self.file_name = file_name
        self.seq_in_file = seq_in_file
        self.label = label if label != 10 else 0
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        
    def get_crop_box(self, crop_adjust=0):
        """
        Get the box used by Image.crop to crop this digit out of the parent.
        """
        return (self.left + crop_adjust, 
                self.top + crop_adjust, 
                self.left + self.width + crop_adjust, 
                self.top + self.height + crop_adjust)
    
    def get_padding(self, to_width, to_height):
        """
        Returns a tuple with the padding required by ImageOps.expand to center
        the digit image in an input block with the given shape. Note that this
        is for the ImageOps.extend() method and returns the pads in this order:
        left, top, right, bottom. Other padders use a different
        convention.
        """
        delta_w = to_width - self.width
        delta_h = to_height - self.height
        return(delta_w//2, delta_h//2, delta_w - (delta_w//2), delta_h - (delta_h//2))
        
    def image_size(self):
        return (self.width, self.height)