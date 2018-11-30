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
from bill_nets import ConvNet

from svhnpickletypes import SvhnDigit
from svhndatasets import SvhnDigitsDataset

import pickle

# For heatmap
import matplotlib.pyplot as plt

# For classification report
from sklearn.metrics import classification_report

# Identify the model to evaluate
STORED_MODEL = os.path.join("results", "bill_net1_1118_195850.pkl")

IMAGE_SIZE = (46,46)
CHANNELS = 1
INPUT_SIZE = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1]) 
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 1000
learning_rate = .001

FORCE_CPU = True

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
# =============================================================================
# Load training and test data
# =============================================================================

# Define a transformation that converts each image to a tensor and normalizes
# each channel
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(IMAGE_SIZE),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS)])

DATA_DIR = os.path.join("..", "..", "data")

# Open the train pickle"
print("Reading pickles")
#with open(os.path.join(DATA_DIR, "train_digit_data.pkl"), 'rb') as f:
#    train_data = pickle.load(f)
with open(os.path.join(DATA_DIR, "test_digit_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading pickles.")

#train_set = SvhnDigitsDataset(train_data, transform=transform)
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = SvhnDigitsDataset(test_data, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Instantiate a model
net = ConvNet(num_classes, CHANNELS, IMAGE_SIZE).to(device=run_device)
print(net)
total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

# Load weights
net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
print("Loading model from: ", STORED_MODEL)

total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)
   
#%%
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
            
        # Accumlate for error metrics
        all_labels += [int(x) for x in labels]
        all_predicted += [int(x) for x in predicted]
        
        if i>4:
            break
         
    print("Done predicting reading test data.")
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

   
#%%
def imshowax(ax, img):
    #img = img / 2 + 0.5
    if type(img) == torch.Tensor:
        npimg = img.numpy()
    else:
        npimg = img
        
    #ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.imshow(npimg, cmap='Greys_r')
    ax.tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
                   labelbottom=False, labelleft=False)
    
    
def show_pred_loop():
    pass
#%%
    normalizer = nn.Softmax(dim=1)
    classes = [str(x) for x in range(10)]
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    
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
            
            imagev = Variable(image).to(device=run_device).view(1,1,46,46)
            imagev.requires_grad_(True)
            
            outputs = net(imagev)
            softmaxed = normalizer(outputs)[0]
            
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            pclass = int(predicted)
            
            # Get grads of input for predicted output
            last_grad = [0] * len(classes)
            #last_grad[pclass] = 1
            other_grads = []
            pred_grads = []
            wavg = np.zeros((net.image_size[0],net.image_size[1]))
            weightsum = 0
            
            for idx in range(len(classes)):
                last_grad = [0] * len(classes)
                last_grad[idx] = 1
                outputs = net(imagev)
                outputs.backward(torch.Tensor(last_grad).view(1,-1), retain_graph=True)
                thesegrads = imagev.grad[0,0].numpy().copy()
                # normalize
                thesegrads = (thesegrads - thesegrads.min()) / (thesegrads.max() - thesegrads.min())
                
                if idx == pclass:
                    pred_grads = thesegrads
                else:
                    other_grads.append(thesegrads)
                    wavg += (1-float(softmaxed[idx])) * thesegrads
                    weightsum += 1-float(softmaxed[idx])
                    
                #all_grads.append((thesegrads - thesegrads.min()) / (thesegrads.max() - thesegrads.min()) )
                
                imagev.grad.data.zero_()
            
            wavg = wavg / (weightsum if weightsum != 0 else 1)
            
            #loss = criterion(outputs, predicted)
            #loss.backward()
           
            #print("Grads", imagev.grad)
            
#            tally= torch.zeros(net.image_size[0],net.image_size[1], device=run_device)
#            tally.requires_grad_(False)
#           
#            # Determine the contribution of each pixel
#            for row in range(net.image_size[0]):
#                print ("Tally row: {0}".format(row))
#                for col in range(net.image_size[1]):
#                    trial_in= torch.ones(net.image_size[0],net.image_size[1], device=run_device)
#                    trial_in.requires_grad_(False)
#                    trial_in = trial_in * 0.5
#                    trial_in[row, col] = imagev[0, 0, row, col]
#                    
#                    # Run the network
#                    trial_out = net(trial_in.view(1,1,net.image_size[0],net.image_size[1]))
#                    tally[row, col] = trial_out[0, predicted]
     
            f, ax = plt.subplots(1, 6, figsize=(10,3.5))
            f.suptitle("Actual: {0} Predicted: {1}".
                   format(classes[label], classes[pclass]))
            
            ax[0].imshow(test_data[image_idx].digit_image)
            ax[0].set_xlabel("Original Image for digit")
            
            imshowax(ax[1], imagev.detach().view(net.image_size[0], net.image_size[1]))
            
            ax[1].set_xlabel("Image {0}".format(image_idx))
            y_pos = np.arange(len(classes))
            ax[2].set_yticks(y_pos)
            ax[2].set_yticklabels(classes, fontsize=8)
            ax[2].set_xlabel("Confidence of class prediction")
            
            
            ax[2].barh(y_pos, softmaxed, align='center',
                    color='blue')
            ax[2].invert_yaxis()
            
            #imshowax(ax[3], tally.detach())
            #ax[3].set_xlabel("Contribution of each pixel")
            #x = imagev.grad[0,0]
            #predg = all_grads[pclass]
            
            # Remove pred
            #del all_grads[pclass]
            
            #x = torch.Tensor(predg - np.asarray(all_grads).mean(axis=0))
            x1 = torch.Tensor(pred_grads) #- np.asarray(other_grads).mean(axis=0))
            #x2 = torch.Tensor(pred_grads - wavg)
            x2 = x1 < 0.35
            x3 = (x2).float() * imagev.view(net.image_size[0], net.image_size[1])
            
            imshowax(ax[3], x1.detach())
            ax[3].set_xlabel("x1")
            
            imshowax(ax[4], x2.detach())
            ax[4].set_xlabel("x2")
            
            imshowax(ax[5], x3.detach())
            ax[5].set_xlabel("x3")
            
            plt.show()


            # make a collection of points selected by the gradients
            
            
#%%  
show_pred_loop()
# =============================================================================
# MAIN -- show the matrix for the loaded model
# =============================================================================
if __name__ == "__main__":
    
    pass

    # Confusion Matrix
    # c_matrix = show_evaluation_metrics(net, run_device, test_loader)