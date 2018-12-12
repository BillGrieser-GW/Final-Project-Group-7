# # Convolutional Neural Network -- Tensorflow
# ### The SVHN dataset (32-by-32 images)
import os
import time
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

#('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size

#print("Tensorflow version", tf.__version__)

# loading the greyscale images created in ``data_preprocess.py``

# Open the file as readonly
h5f = h5py.File('SVHN_grey.h5', 'r')

# Load the training, test and validation set and close it.
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
X_val = h5f['X_val'][:]
y_val = h5f['y_val'][:]

h5f.close()

print('Training set', X_train.shape, y_train.shape)
print('Validation set', X_val.shape, y_val.shape)
print('Test set', X_test.shape, y_test.shape)

# We know that SVHN images have 32 pixels in each dimension
img_size = X_train.shape[1]

# Greyscale images only have 1 color channel
num_channels = X_train.shape[-1]

# Number of classes, one class for each of 10 digits
num_classes = y_train.shape[1]

# Applying mean subtraction and normalization to our images.
train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)

# Subtract it equally from all splits
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean)  / train_std
X_val = (train_mean - X_val) / train_std

def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    fig, axes = plt.subplots(nrows, ncols)
    rs = np.random.choice(images.shape[0], nrows*ncols)
    for i, ax in zip(rs, axes.flat): 
        if cls_pred is None:
            title = "True: {0}".format(np.argmax(cls_true[i]))
        else:
            title = "True: {0}, Pred: {1}".format(np.argmax(cls_true[i]), cls_pred[i])  
        ax.imshow(images[i,:,:,0], cmap='binary')
        # Annotate the image
        ax.set_title(title)
        # Do not overlay a grid
        ax.set_xticks([])
        ax.set_yticks([])        
# Plot 2 rows with 9 images each from the training set
# plot_images(X_train, 2, 9, y_train);

# functions for creating new variables
def conv_weight_variable(layer_name, shape):
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def fc_weight_variable(layer_name, shape):
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))


# ### Helper function for stacking CONV-RELU layers followed by an optional POOL layer
# 
# This function creates a new convolutional layer in the computational graph for TensorFlow. 
# The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point it is common to transition to fully connected layers.
# INPUT > [[CONV -> RELU]*N -> POOL?]M -> [FC -> RELU]*K -> FC```
def conv_layer(input,               
                layer_name,         
                num_input_channels, 
                filter_size,        
                num_filters,        
                pooling=True):      
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = conv_weight_variable(layer_name, shape=shape)
    biases = bias_variable(shape=[num_filters])

    # Create the TensorFlow operation for convolution
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME') # with zero padding
    layer += biases
    layer = tf.nn.relu(layer)
    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    return layer, weights
# function for reshaping the CONV layers to FC layers
# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the convolution layers, 
# so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

# Function for stacking FC-RELU layers
# This function creates a new fully-connected layer in the computational graph for TensorFlow. 
# Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.

def fc_layer(input,        # The previous layer
             layer_name,   # The layer name
             num_inputs,   # Num. inputs from prev. layer
             num_outputs,  # Num. outputs
             relu=True):   # Use RELU?
    weights = fc_weight_variable(layer_name, shape=[num_inputs, num_outputs])
    biases = bias_variable(shape=[num_outputs])
    layer = tf.matmul(input, weights) + biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer

#################################################################### Tensorflow Model #####################################################################

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 256            # Number of neurons in fully-connected layer.

# ### Placeholder Variables
# Placeholder variables serve as the input to the graph that we may change each time we execute the graph. 
# First we define the placeholder variable for the input images. This allows us to change the images that are input to the TensorFlow graph. 
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
keep_prob = tf.placeholder(tf.float32)

## ConvNet Architecture 
# #### Convolutional Layer 1
# Create the first convolutional layer. It takes x as input and creates num_filters1 different filters, each having width and height equal to filter_size1. 
conv_1, w_c1 = conv_layer(input=x,
                          layer_name="conv_1",
                          num_input_channels=num_channels,
                          filter_size=filter_size1,
                          num_filters=num_filters1, pooling=True)
# #### Convolutional Layer 2
# Create the second convolutional layer, which takes as input the output from the first convolutional layer. 
#The number of input channels corresponds to the number of filters in the first convolutional layer. Finally we wish to down-sample the image so it is half the size by using 2x2 max-pooling.
conv_2, w_c2 = conv_layer(input=conv_1,
                          layer_name="conv_2",
                          num_input_channels=num_filters1,
                          filter_size=filter_size2,
                          num_filters=num_filters2,
                          pooling=True)
# Apply dropout after the pooling operation
dropout = tf.nn.dropout(conv_2, keep_prob)

# #### Flatten Layer
# The convolutional layers output 4-dim tensors. We now wish to use these as input in a fully-connected network, which requires for the tensors to be reshaped or flattened to 2-dim tensors.
layer_flat, num_features = flatten_layer(dropout)

# #### Fully-Connected Layer 1
# Add a fully-connected layer to the network. The input is the flattened layer from the previous convolution.
fc_1 = fc_layer(input=layer_flat,
                layer_name="fc_1",
                num_inputs=num_features,
                num_outputs=fc_size,
                relu=True)

# #### Fully-Connected Layer 2
 # Add another fully-connected layer that outputs vectors of length 10 for determining which of the 10 classes the input image belongs to.
fc_2 = fc_layer(input=fc_1,
                layer_name="fc_2",
                num_inputs=fc_size,
                num_outputs=num_classes,
                relu=False)
# #### Predicted Class 
# The second fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes.
y_pred = tf.nn.softmax(fc_2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Calcualte the cross-entropy and take avg for all the image classifications.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)


# ### Optimization Method
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)
# Construct a new Adam optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)


# ### Evaluation Metric
# To evaluate the performance of our Convolutional Network we calculate the average accuracy across all samples
# Predicted class equals the true class of each image?
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


############################################################# Execution Phase ###############################################################
# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.initialize_all_variables())

# In order to save the variables of the neural network, we now create a so-called Saver-object which is used for storing and retrieving all the variables of the TensorFlow graph. The saved files are often called checkpoints because they may be written at regular intervals during optimization.
# This is the directory used for saving and retrieving the data.
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'svhn_single_greyscale')
#saver.restore(sess=session, save_path=save_path)
# There are 600,000 images in the training-set. It takes a long time to calculate the gradient of the model using all these images. 
# We therefore only use a small batch of images in each iteration of the optimizer. Additionally we must select a keep probability for our dropout tensor. 
# The value of p=0.5 is a reasonable default, but this can be tuned on validation data. 
# Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well.
 #Number of training samples in each iteration
batch_size = 64
dropout = 0.5
total_iterations = 0

def optimize(num_iterations, display_step):
    global total_iterations
    start_time = time.time()
    for step in range(num_iterations):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict_train = {x: batch_data, y_true: batch_labels, keep_prob: dropout}
        # Run the optimizer using this batch of training data.
        session.run(optimizer, feed_dict=feed_dict_train)
        if step % display_step == 0:
            # Calculate the accuracy on the training-set.
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Minibatch accuracy at step %d: %.4f" % (step, batch_acc))
            # Calculate the accuracy on the validation-set
            validation_acc = session.run(accuracy, {x: X_val, y_true: y_val, keep_prob: 1.0})
            print("Validation accuracy: %.4f" % validation_acc)
    total_iterations += num_iterations
    time_diff = time.time() - start_time
    # Calculate the accuracy on the test-set
    test_accuracy = session.run(accuracy, {x: X_test, y_true: y_test, keep_prob: 1.0})
    print("Test accuracy: %.4f" % test_accuracy)
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))

optimize(num_iterations=6000, display_step=500)
saver.save(sess=session, save_path=save_path)

''