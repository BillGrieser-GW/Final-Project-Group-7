# =============================================================================
# Question 10
#
# Load a model and show its confusion matrix versus the test data
#
# =============================================================================
import sys

# Allow imports from parent dir
sys.path.insert(0,"..")

import os
import torch
import torchvision.transforms as transforms
import numpy as np
import numpy.ma as ma

import torch.nn as nn
from torch.autograd import Variable
import predictor_nets
from plot_helpers import imshowax

from svhnpickletypes import SvhnDigit
from svhndatasets import SvhnDigitsDataset

import pickle

# For heatmap
import matplotlib.pyplot as plt

# For classification report
from sklearn.metrics import classification_report

# Identify the model to evaluate

STORED_MODEL = os.path.join("results", "basis_runs", "train_predictor_1208_230417.pkl")
DATA_DIR = os.path.join("..", "..", "data")

IMAGE_SIZE = (40,40)
CHANNELS = 1
INPUT_SIZE = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1])
num_classes = 10
batch_size = 1000

FORCE_CPU = True

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
# =============================================================================
# Load test data
# =============================================================================
# Open the train pickle"
print("Reading pickles")

with open(os.path.join(DATA_DIR, "test_digit_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading pickles.")

# Instantiate a model
net = predictor_nets.ConvNet48(num_classes, CHANNELS, IMAGE_SIZE).to(device=run_device)
print(net)
total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

# Load weights
net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
print("Loading model from: ", STORED_MODEL)

# Turn test data into something the model can use
test_set = SvhnDigitsDataset(test_data, transform=net.get_transformer())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# =============================================================================
# Display results summary
# =============================================================================
def show_evaluation_metrics(net, run_device, test_loader):
    
    classes = [str(x) for x in range(10)]
    # x is predicted
    # y is actual
    c_matrix = np.zeros((net.num_classes, net.num_classes), dtype=int)
    
    all_labels = []
    all_predicted = []
    
    for i, data in enumerate(test_loader):
        print("Reading batch {0}".format(i))
        
        images, labels = data
        images = Variable(images).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        
        for idx in range(len(predicted)):
            c_matrix[labels[idx], predicted[idx]] += 1
            
        # Accumulate for error metrics
        all_labels += [int(x) for x in labels]
        all_predicted += [int(x) for x in predicted]
        
        if i>3:
            break
         
    print("Done predicting test data.")
    # Draw the confusion matrix

    # This uses Seaborn, which gives a nice plot; however, our cloud instances
    # don't have the latest version of matplotlib and this code displays the
    # confusion matrix without the counts instead of including them.
    # import seaborn as sns
    # fig, ax = plt.subplots(figsize=(9, 6))
    # plt.title("Confusion Matrix", fontsize=16)
    # sns.set(font_scale=1.0)  # Label size
    # sns.heatmap(c_matrix, annot=True, annot_kws={"size": 14}, robust=True, fmt='d', \
    #            linecolor='gray', linewidths=0.5, square=False, cbar=True, cmap='Blues',
    #            xticklabels=classes, yticklabels=classes)
    # plt.ylabel('Actual Labels', fontsize=14)
    # plt.xlabel('Predicted Labels', fontsize=14)
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.show()
    
    # This draws a confusion matrix using just matplotlib
    fig, ax = plt.subplots(figsize=(8, 7))
   
    # Draw the grid squares and color them based on the value of the underlying measure
    ax.matshow(c_matrix, cmap=plt.cm.Blues, alpha=0.6)
    
    # Set the tick labels size
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
        
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            plt.text(x=j, y=i, s="{0}".format(c_matrix[i,j]), 
                      va='center', ha='center', fontdict={'fontsize':14})
            
    plt.ylabel('Actual Labels', fontsize=12) 
    plt.xlabel('Predicted Labels', fontsize=12)
    tick_marks = np.arange(10)
    #ax.set_ticks_position(position='bottom')
    plt.grid(b=False)
    plt.xticks(tick_marks, [str(x) for x in range(10)], rotation=0, fontsize=10)
    plt.yticks(tick_marks, [str(x) for x in range(10)], fontsize=10)       
    plt.suptitle("Confusion Matrix", fontdict={'fontsize':16})
    plt.show()
    
    # Print Classification Report
    print("\nClassification Report\n")
    print(classification_report(all_labels, all_predicted, target_names=classes))
    
    # IN CASE OF EMERGENCY: Uncomment to see a text-only CM
    #print("\nConfusion matrix (non-graphically)\n")
    #print(c_matrix)
    return c_matrix

def show_pred_loop():

    normalizer = nn.Softmax(dim=1)
    classes = [str(x) for x in range(10)]

    while True:
        image_idx = input("Enter an index from 0 to {0} from the test data (q to quit): ".format(len(test_set)))
        
        try:
            if image_idx.lower() == 'q':
                break
            
            image_idx = int(image_idx)
            
        except:
            print("Bad input -- assuimg 0")
            image_idx = 0
            
        if image_idx >= 0 and image_idx < len(test_set):
            
            # Get the image & label
            image, label = test_set[image_idx]
            imagev = Variable(image).to(device=run_device).view(1,1,IMAGE_SIZE[0],IMAGE_SIZE[1])
            imagev.requires_grad_(True)
            
            outputs = net(imagev)
            softmaxed = normalizer(outputs)[0]
            
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            pclass = int(predicted)
            
            # Get grads of input for predicted output
            last_grad = [0] * len(classes)
            last_grad[pclass] = 1
            outputs.backward(torch.Tensor(last_grad).view(1, -1))
            pred_grads = imagev.grad[0, 0].numpy().copy()

            f, ax = plt.subplots(1, 4, figsize=(10,8))
            f.suptitle("Image: {2} Parent: {3}\nActual: {0} Predicted: {1}".
                   format(classes[label], classes[pclass], image_idx, test_data[image_idx].data.file_name))

            imshowax(ax[0], test_data[image_idx].digit_image)
            ax[0].set_xlabel("Original Image for digit")
            
            imshowax(ax[1], imagev.detach().view(net.image_size[0], net.image_size[1]))
            ax[1].set_xlabel("Transformed Image")

            y_pos = np.arange(len(classes))
            ax[2].set_yticks(y_pos)
            ax[2].set_yticklabels(classes, fontsize=8)
            ax[2].set_xlim([0, 1])
            ax[2].set_xlabel("Confidence of\nclass prediction")

            ax[2].barh(y_pos, softmaxed, align='center',
                    color='blue')
            ax[2].invert_yaxis()

            x1 = torch.Tensor(pred_grads)
            imshowax(ax[3], x1.detach())
            ax[3].set_xlabel("Gradients of input\nwrt Output by pixel")

            plt.show()

# =============================================================================
# MAIN -- show the matrix for the loaded model
# =============================================================================
if __name__ == "__main__":

    # Confusion Matrix
    #c_matrix = show_evaluation_metrics(net, run_device, test_loader)

    show_pred_loop()