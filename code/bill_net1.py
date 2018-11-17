# =============================================================================
# net 1
#
# Pytorch, using GPU iof available
#
# =============================================================================

# --------------------------------------------------------------------------------------------
import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import datetime

from svhnpickletypes import SvhnDigit
from svhndatasets import SvhnDigitsDataset
import pickle

# --------------------------------------------------------------------------------------------
# Choose the right values for x.

#
# Initially try a network with a single hidden layer of 500 neurons
# and a moderate number of neurons and learning rate
# 

IMAGE_SIZE = (32,32)
CHANNELS = 1
input_size = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1]) 
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 250
learning_rate = .01

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
transform = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize(IMAGE_SIZE),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
# --------------------------------------------------------------------------------------------

DATA_DIR = os.path.join("..", "data")

# Open the train pickle"
print("Reading pickles")
with open(os.path.join(DATA_DIR, "train_digit_data.pkl"), 'rb') as f:
    train_data = pickle.load(f)
with open(os.path.join(DATA_DIR, "test_digit_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading pickles.")

train_set = SvhnDigitsDataset(train_data, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = SvhnDigitsDataset(test_data, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#%%
# =============================================================================
# Set up the network
# =============================================================================
# --------------------------------------------------------------------------------------------
# Choose the right argument for xx
#
# Define a model class. This uses purelin() as the second-layer
# transfer function
class FlatNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FlatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out
    
# --------------------------------------------------------------------------------------------
def get_total_parms(module):
    
    total_parms = 0
    # Find the total number of parameters being trained
    for t in module.state_dict().values():
        total_parms += np.prod(t.shape)
    return total_parms
        
# Fixed manual seed
torch.manual_seed(267)

# Choose the right argument for x
        
# Instantiate a model
net = ConvNet(input_size, hidden_size, num_classes).to(device=run_device)
print(net)
total_net_parms = get_total_parms(net)
print ("Total trainable parameters:", total_net_parms)

# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# =============================================================================
# Make a run
# =============================================================================
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data
        #images= images.view(-1, CHANNELS * 32 * 32)
        # Put the images and labels in tensors on the run device
        images= Variable(images).to(device=run_device)
        labels= Variable(labels).to(device=run_device)
        optimizer.zero_grad()
        outputs = net(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.data.item()))
            
# =============================================================================
# Display results summary
# =============================================================================
#%%
correct = 0
total = 0
for images, labels in test_loader:
    
    #imagesv = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
    imagesv = Variable(images).to(device=run_device)
    outputs = net(imagesv)
    
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    
    # Bring the predicted values to the CPU to compare against the labels
    correct += (predicted.cpu() == labels).sum()

print('Accuracy of the network on {0} test images: {1:0.1f}%'.format(total, float(100 * correct) / total))
#%%
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

for images, labels in test_loader:
    #imagesv = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
    imagesv = Variable(images).to(device=run_device)
    outputs = net(imagesv)
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted.cpu() == labels)
    for i in range(len(c)):
        label = labels[i]
        class_correct[label] += c[i].int()
        class_total[label] += 1

# --------------------------------------------------------------------------------------------
for i in range(num_classes):
    print('Accuracy of {0:3d} : {1:0.1f}%'.format(i, float(100 * class_correct[i]) / class_total[i]))
#%%
# =============================================================================
# Save results to a file for comparison purposes.
# =============================================================================
now = datetime.datetime.now()
suffix = "_" + now.strftime('%m%d_%H%M%S')

if not os.path.exists('results'):
    os.makedirs('results')
    
run_base='console'

if sys.argv[0] != '':
    run_base = os.path.basename(sys.argv[0])
    run_base = os.path.join(os.path.splitext(run_base)[0])
    
run_base=os.path.join('results', run_base)

# Save run artifacts
torch.save(net.state_dict(), run_base + suffix + '.pkl')

with open(run_base + suffix + '_results.txt', 'w') as rfile:
    rfile.write(str(net))
    rfile.write('\n\n')
    rfile.write("Total network weights + biases: {0}\n".format(total_net_parms))
    rfile.write("Epochs: {0}\n".format(num_epochs))
    rfile.write("Learning rate: {0}\n".format(learning_rate))
    rfile.write("Batch Size: {0}\n".format(batch_size))
    rfile.write("Final loss: {0:0.4f}\n".format(loss.data.item()))
    rfile.write("Run device: {0}\n".format(run_device))
    rfile.write('\n')
    rfile.write('Accuracy of the network on the {0:d} test images: {1:0.1f}%\n'.format(total, float(100 * correct) / total))
    rfile.write('\n')
    
    for i in range(num_classes):
        rfile.write('Accuracy of {0:3s} : {1:0.1f}%\n'.format(str(i), float(100 * class_correct[i]) / class_total[i]))
    
