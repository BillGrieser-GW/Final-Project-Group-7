# =============================================================================
# Question 4
#
# Try several archictures of neurons in layers. The code to build a 
# network is modified to accept a variable number of layers, and then
# several runs are performed.
#
# =============================================================================

# --------------------------------------------------------------------------------------------
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# Choose the right values for x.

# Identify the model to evaluate
STORED_MODEL = os.path.join("results", "best_model.pkl")

CHANNELS = 3
input_size = (CHANNELS * 32 * 32) # 3 color 32x32 images
hidden_size = [(1500,), ]
optimizers = [torch.optim.Adagrad]
transfer_functions = [nn.ReLU]
dropout= [0.5]
num_classes = 10
num_epochs = 0
batch_size = 32
learning_rate = .005

FORCE_CPU = False

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
    
# =============================================================================
# Load training and test data
# =============================================================================
# --------------------------------------------------------------------------------------------
# Define a transformation that converts each image to a tensor and normalizes
# each channel
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------

DATA_ROOT = os.path.join("..", "data_cifar")

test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------------------------------------------------------------------------------------------
#
# Helper function
#

def get_total_parms(module):
    """Get the total number of trainable parameters in a network"""
    total_parms = 0
    for t in module.state_dict().values():
        total_parms += np.prod(t.shape)
    return total_parms

# =============================================================================
# Model Class
# 
# Define a model class that takes a variable number of hidden layer sizes
# and contructs a network to match. The network uses ReLu as the transfer
# function in each hidden layer. It uses purelin on the output layer.
#
# =============================================================================
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, 
                 transfer_function=nn.ReLU, dropout=0.0):
        super(Net, self).__init__()
        
        last_in = input_size
        self.hidden_layers = []
        
        for idx, sz in enumerate(hidden_size):
            new_module = nn.Linear(last_in, sz)
            self.add_module("layer_{0:02d}".format(idx+1), new_module)
            self.hidden_layers.append((new_module, transfer_function()))    
            last_in = sz
            
        self.dropout_layer=nn.Dropout(dropout)
        
        # Add the output layer (with an implied purelin activation)
        self.output_layer = nn.Linear(last_in, num_classes)

    def forward(self, x):
        out = x
        for layer, transfer in self.hidden_layers:
            out = layer(out)
            out = transfer(out)
        
        # Dropout
        out = self.dropout_layer(out)
        
        # Output layer
        out = self.output_layer(out)
        return out

# =============================================================================
# Function to make a model and load it
# =============================================================================
def make_model(this_hidden_size, run_device, 
                      transfer_function=nn.ReLU, dropout=0.0):

    # Fixed manual seed
    torch.manual_seed(267)
    start_time = time.time()
    
    # Instantiate a model
    net = Net(input_size, this_hidden_size, num_classes, transfer_function=transfer_function,
              dropout=dropout).to(device=run_device)
    print(net)
    net.train(True)
    
    net.load_state_dict(torch.load(STORED_MODEL, map_location=run_device))
    print("Loading from: ", STORED_MODEL)
    
    total_net_parms = get_total_parms(net)
    print ("Total trainable parameters:", total_net_parms)
   
    return net, time.time() - start_time
      
# =============================================================================
# Display results 
# =============================================================================
def imshowax(ax, img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.tick_params(axis='both', which = 'both', bottom=False, left=False, tick1On=False, tick2On=False,
                   labelbottom=False, labelleft=False)
    
# =============================================================================
# MAIN -- Accept input from the user for test data and results to display
# =============================================================================
if __name__ == "__main__":

    # Make a model
    net, duration = make_model(hidden_size[0],
                              run_device=run_device, 
                              transfer_function=transfer_functions[0],
                              dropout=dropout[0])
                    
    # Get some test data
    for images, labels in test_loader:
        images = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        outputs = outputs.cpu()
        images = images.cpu()
    
    correct = 0
    # How many are correct?
    for idx in range(len(labels)):
        if predicted[idx] == labels[idx]:
            correct += 1
            
    print("{0} correct out of {1} total test samples.".format(correct, len(labels)))

#%%
    normalizer = nn.Softmax(dim=1)
    while True:
        image_idx = input("Enter an index from 0 to 9999 from the test data (q to quit): ")
        
        try:
            if image_idx.lower() == 'q':
                break
            
            image_idx = int(image_idx)
            
        except:
            print("Bad input -- assuimg 0")
            image_idx = 0
            
        if image_idx >= 0 and image_idx < 10000:
            #imshow(images[image_idx].reshape(3,32,32), title="Actual: {0} Predicted: {1}".
            #       format(classes[labels[image_idx]], classes[int(predicted[image_idx])]))
            
            f, ax = plt.subplots(1, 2, figsize=(9.5,3.5))
            f.suptitle("Actual: {0} Predicted: {1}".
                   format(classes[labels[image_idx]], classes[int(predicted[image_idx])]))
            imshowax(ax[0], images[image_idx].reshape(3,32,32))
            ax[0].set_xlabel("Image {0}".format(image_idx))
            y_pos = np.arange(len(classes))
            ax[1].set_yticks(y_pos)
            ax[1].set_yticklabels(classes, fontsize=8)
            ax[1].set_xlabel("Confidence of class prediction")
            softmaxed = normalizer(outputs[image_idx:image_idx+1])[0]
            
            ax[1].barh(y_pos, softmaxed, align='center',
                    color='blue')
            ax[1].invert_yaxis()
            plt.show()

        
        