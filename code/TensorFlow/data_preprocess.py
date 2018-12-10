# ** SVHN Dataset Preprocessing the 32 x 32 image datset **

# Pre-processing the 32-by-32 images from the SVHN dataset centered around a single digit. In this dataset all digits have been resized to a fixed resolution of 32-by-32 pixels.
# The original character bounding boxes are extended in the appropriate dimension to become square windows, so that resizing them to 32-by-32 pixels does not introduce aspect ratio distortions
# 
#    Steps:
#    * Create a Startified 13% of data in Validation Set
#    * Converting the Label 10's to 0's
#    * Greyscale conversion of image(data) for easy computation
#    * Normalization of data 


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from skimage import color
from skimage import io
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (16.0, 4.0)


## Loading Data .....
# Reading the .MAT files

def load_data(path):
    data = loadmat(path)
    return data['X'], data['y']

X_train, y_train = load_data('Data/train_32x32.mat')
X_test, y_test = load_data('Data/test_32x32.mat')

#print("Training Set", X_train.shape, y_train.shape)
#print("Test Set", X_test.shape, y_test.shape)


# Transposing the the train and test data by converting it from  (width, height, channels, size) -> (size, width, height, channels)

X_train, y_train = X_train.transpose((3,0,1,2)), y_train[:,0]
X_test, y_test = X_test.transpose((3,0,1,2)), y_test[:,0]

print("Training Set", X_train.shape)
print("Test Set", X_test.shape, "\n")


# Calculate the total number of images
num_images = X_train.shape[0] + X_test.shape[0]

print("Total Number of Images", num_images)

# ` Plotting Function for fig in n rows X m columns 
# can be used for grayscale and RGB both`

def plot_images(img, labels, nrows, ncols):
    
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
            plt.show()

        else:
            ax.imshow(img[i,:,:,0])
            plt.show()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])
    #plt.show()
# Plot some training set images
# plot_images(X_train, y_train, 2, 8)

# Plot some test set images
# plot_images(X_test, y_test, 2, 8)

## To check unique labels
# print(np.unique(y_train))

## Plotting Distribution of Data
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)
ax1.hist(y_train, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(1, 10)
ax2.hist(y_test, color='g', bins=10)
ax2.set_title("Test set")
fig.tight_layout()
plt.show()
# All distributions have a positive skew, meaning that we have an underweight of higher values.

## Converting Label 10 -> 0

y_train[y_train == 10] = 0
y_test[y_test == 10] = 0
print(np.unique(y_train))

## Splitting the Training to Train + Validation
# `Splitting to 13% in Val Set as it gives around 9500 data having min. of 800 instances of each class`

# `Using random state to regenrate the whole Dataset in re-run`
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.13, random_state=7)

## Visualize New distribution
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)
ax1.hist(y_train, bins=10)
ax1.set_title("Training set")
ax1.set_xlim(1, 10)
ax2.hist(y_val, color='g', bins=10)
ax2.set_title("Validation set")
fig.tight_layout()
plt.show()

# `Data in each Set`
# y_train.shape, y_val.shape, y_test.shape

# ## Grayscale Conversion 

# To speed up our experiments we will convert our images from RGB to Grayscale, which grately reduces the amount of data we will have to process. 
#  ** Y = 0.2990R + 0.5870G + 0.1140B **
# Here is a simple function that helps us print the size of a numpy array in a human readable format.
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

# Converting to Float for numpy computation
train_greyscale = rgb2gray(X_train).astype(np.float32)
test_greyscale = rgb2gray(X_test).astype(np.float32)
val_greyscale = rgb2gray(X_val).astype(np.float32)


'''print("Training Set", train_greyscale.shape)
print("Validation Set", val_greyscale.shape)
print("Test Set", test_greyscale.shape)
print('')'''

# Removing RGB train, test, val set  to reduce RAM Storage occupied by them
del X_train, X_test, X_val


# ### Ploting the Grayscale Image

# Before Normalization
# plot_images(train_greyscale, y_train, 1, 10)


## Normalization

# Calculate the mean and std on the training data
train_mean = np.mean(train_greyscale, axis=0)
train_std = np.std(train_greyscale, axis=0)

# Subtract it equally from all splits
train_greyscale_norm = (train_greyscale - train_mean) / train_std
test_greyscale_norm = (test_greyscale - train_mean)  / train_std
val_greyscale_norm = (val_greyscale - train_mean) / train_std

# `Plotting After Normalization`
# plot_images(train_greyscale_norm, y_train, 1, 10)


# ### One Hot Label Encoding
from sklearn.preprocessing import OneHotEncoder
 
# Fit the OneHotEncoder
enc = OneHotEncoder().fit(y_train.reshape(-1, 1))

# Transform the label values to a one-hot-encoding scheme
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()

#print("Training set", y_train.shape)
#print("Validation set", y_val.shape)
#print("Test set", y_test.shape)

## Storing Data to Disk
import h5py

# Create file
h5f = h5py.File('SVHN_grey.h5', 'w')

# Store the datasets
h5f.create_dataset('X_train', data=train_greyscale_norm)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=test_greyscale_norm)
h5f.create_dataset('y_test', data=y_test)
h5f.create_dataset('X_val', data=val_greyscale_norm)
h5f.create_dataset('y_val', data=y_val)

# Close the file
h5f.close()

