
## Convolutional Neural Network
## The SVHN dataset (32-by-32 images)

import os
import time
import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (16.0, 4.0) # Set default figure size
#print("Tensorflow version", tf.__version__)

h5f = h5py.File('SVHN_grey.h5', 'r')

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

img_size = X_train.shape[1]
# Greyscale images only have 1 color channel
num_channels = X_train.shape[-1]
# Number of classes, one class for each of 10 digits
num_classes = y_train.shape[1]

####################################################################  PREPROCESSING  ##########################################################################

train_mean = np.mean(X_train, axis=0)
train_std = np.std(X_train, axis=0)
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
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])        

####################################################  BUILDING A TENSORFLOW COMPUTATIONAL GRAPH  ################################################################

def conv_weight_variable(layer_name, shape):
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())
def fc_weight_variable(layer_name, shape):
    return tf.get_variable(layer_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape):
    return tf.Variable(tf.constant(0.0, shape=shape))

# function for stacking CONV-RELU layers followed by an optional POOL layer

def conv_layer(input,               # The previous layer
                layer_name,         # Layer name
                num_input_channels, # Num. channels in prev. layer
                filter_size,        # Width and height of each filter
                num_filters,        # Number of filters
                pooling=True):      # Use 2x2 max-pooling
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = conv_weight_variable(layer_name, shape=shape)
    biases = bias_variable(shape=[num_filters])
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
# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

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

############################################################   TENSORFLOW MODEL  ######################################################################################

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 32         # There are 16 of these filters.
# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 64         # There are 36 of these filters.
# Fully-connected layer.
fc_size = 256            # Number of neurons in fully-connected layer.

# ### Placeholder Variables
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# To reduce overfitting, i applied dropout after the pooling layer. Created a placeholder for the probability that a neuron's output is kept during dropout. 
# This allows us to turn dropout on during training, and turn it off during testing. 
keep_prob = tf.placeholder(tf.float32)

## ConvNet Architecture
# ``` INPUT -> [CONV -> RELU -> POOL -> CONV -> RELU -> POOL] -> DROPOUT -> [FC -> RELU] -> FC -> Softmax
conv_1, w_c1 = conv_layer(input=x,
                          layer_name="conv_1",
                          num_input_channels=num_channels,
                          filter_size=filter_size1,
                          num_filters=num_filters1, pooling=True)

conv_2, w_c2 = conv_layer(input=conv_1,
                          layer_name="conv_2",
                          num_input_channels=num_filters1,
                          filter_size=filter_size2,
                          num_filters=num_filters2,
                          pooling=True)

dropout = tf.nn.dropout(conv_2, keep_prob)
layer_flat, num_features = flatten_layer(dropout)

fc_1 = fc_layer(input=layer_flat,
                layer_name="fc_1",
                num_inputs=num_features,
                num_outputs=fc_size,
                relu=True)

fc_2 = fc_layer(input=fc_1,
                layer_name="fc_2",
                num_inputs=fc_size,
                num_outputs=num_classes,
                relu=False)
# SOFTMAX
# The second fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes. 
y_pred = tf.nn.softmax(fc_2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

## Cost Function/ CROSS ENTROPY
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

## Optimization Method - Adagrad and Exponential decay
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96, staircase=True)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost, global_step=global_step)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################################  EXECUTION PHASE ###################################################################################

## Create TensorFlow Session
session = tf.Session()
session.run(tf.global_variables_initializer())

## Saver
# In order to save the variables of the neural network, we now create a so-called Saver-object which is used for storing and retrieving all the variables of the TensorFlow graph.
saver = tf.train.Saver()
save_dir = 'checkpoints/'

# Create directory if it does not exist
# os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'svhn_single_greyscale')
saver.restore(sess=session, save_path=save_path)

## Executing the learning process
batch_size = 64
dropout = 0.5

total_iterations = 0

def optimize(num_iterations, display_step):
    global total_iterations
    start_time = time.time()
    acc_test = []
    for step in range(num_iterations):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]
        feed_dict_train = {x: batch_data, y_true: batch_labels, keep_prob: dropout}
        session.run(optimizer, feed_dict=feed_dict_train)

        if step % display_step == 0:

            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            print("Minibatch accuracy at step %d: %.4f" % (step, batch_acc))
            
            validation_acc = session.run(accuracy, {x: X_val, y_true: y_val, keep_prob: 1.0})
            print("Validation accuracy: %.4f" % validation_acc)
            acc_test.append(validation_acc)
    total_iterations += num_iterations
    time_diff = time.time() - start_time
    
    # Calculate the accuracy on the test-set
    test_accuracy = session.run(accuracy, {x: X_test, y_true: y_test, keep_prob: 1.0})
    print("Test accuracy: %.4f" % test_accuracy)
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
    print("Test accuracy: %.4f" % test_accuracy)
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
    plt.figure(1)
    plt.plot(acc_test)
    plt.xlabel('Number of iterations')
    plt.ylabel("Test accuracy values")
    plt.title("Test accuracy")
    plt.show()
#optimize(num_iterations=1000, display_step=500)
optimize(num_iterations=50000, display_step=1000)
saver.save(sess=session, save_path=save_path)