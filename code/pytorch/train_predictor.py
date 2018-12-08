# =============================================================================
# Train a network on the SVHN dataset
#
# =============================================================================

import os
import sys
import argparse
import csv

# Allow imports from parent dir
sys.path.insert(0,"..")

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import datetime
import time
from svhndatasets import SvhnDigitsDataset
import pickle
import predictor_nets 

# =============================================================================
# Command Line Args
# =============================================================================
optimizer_args = lambda x: {"SGD": torch.optim.SGD, 
                            "ASGD": torch.optim.ASGD, \
                            "Adagrad": torch.optim.Adagrad}[x]

network_args = lambda x: {"ConvNet32": predictor_nets.ConvNet32, 
                           "ConvNet48": predictor_nets.ConvNet48,
                           "ConvNet32_753": predictor_nets.ConvNet32_753}[x]

# Get command-line arguments
parser = argparse.ArgumentParser(description='Train SVHN predictor.')
parser.add_argument('--batch', type=int, default=32,
                   help='Batch Size')
parser.add_argument('--epochs', type=int, default=2,
                   help='Epochs')
parser.add_argument('--opt', type=optimizer_args, default='SGD', 
                   help='Optimizer (SGD, Adagrad, ASGD)')
parser.add_argument('--net', type=network_args, default='ConvNet32', 
                   help='Network archicture (ConvNet32, ConvNet48, Convnet32_753)')
parser.add_argument('--cpu', action='store_true', 
                   help='Force to CPU even if GPU present')
args = parser.parse_args()

batch_size = args.batch
network_class = args.net
optimizer = args.opt
FORCE_CPU = args.cpu
num_epochs = args.epochs

print("Using optimizer:", optimizer, "\nwith batch size:", batch_size, 
      "and epochs:", num_epochs)

# =============================================================================
# Other setup
# =============================================================================
IMAGE_SIZE = (40,40)
CHANNELS = 1
INPUT_SIZE = (CHANNELS * IMAGE_SIZE[0] * IMAGE_SIZE[1]) 
num_classes = 10
learning_rate = .001

if torch.cuda.is_available() and FORCE_CPU != True:
    print("Using cuda device for Torch")
    run_device = torch.device('cuda')
else:
    print("Using CPU devices for Torch.")
    run_device = torch.device('cpu')
#%%    
# =============================================================================
# Load training and test data
# =============================================================================
DATA_DIR = os.path.join("..", "..", "data")

# Open the train pickle
print("Reading pickles")
with open(os.path.join(DATA_DIR, "train_digit_data.pkl"), 'rb') as f:
    train_data = pickle.load(f)
with open(os.path.join(DATA_DIR, "test_digit_data.pkl"), 'rb') as f:
    test_data = pickle.load(f)
print("Done Reading pickles.")

def get_accuracy(limit=0):
    """
    Helper function to calculate accuracy against the test data
    :param limit: Stop after reaching this numger of test samples; 0 to do all
    :return:
    """
    correct = 0
    total = 0
    net.eval()

    for images, labels in test_loader:

        #imagesv = Variable(images.view(-1, CHANNELS * 32 * 32)).to(device=run_device)
        imagesv = Variable(images).to(device=run_device)
        outputs = net(imagesv)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # Bring the predicted values to the CPU to compare against the labels
        correct += (predicted.cpu() == labels).sum()

        if limit > 0 and total >= limit:
            break

    return total, correct
#%%
# =============================================================================
# Prepare for training; perform the transform required by the network
# =============================================================================
# Fixed manual seed
torch.manual_seed(267)

# Instantiate a model
net = network_class(num_classes, CHANNELS, IMAGE_SIZE ).to(device=run_device)

#net = FlatNet(INPUT_SIZE, hidden_size, num_classes).to(device=run_device)
print(net)
total_net_parms = net.get_total_parms()
print ("Total trainable parameters:", total_net_parms)

train_set = SvhnDigitsDataset(train_data, transform=net.get_transformer())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = SvhnDigitsDataset(test_data, net.get_transformer())
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#%%
# =============================================================================
# Make a run
# =============================================================================
# Define the loss and optimization functions to use
criterion = nn.CrossEntropyLoss() 
optimizer = optimizer(net.parameters(), lr=learning_rate)
dtype = torch.float32

run_history = []

start_time = time.time()
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i, data in enumerate(train_loader):

        net.train()
        
        images, labels = data
        
        # Put the images and labels in tensors on the run device
        images= Variable(images).to(device=run_device)
        labels= Variable(labels).to(device=run_device)
        
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item()

        if ((i + 1) % 100 == 0) or ((i+1) == (len(train_set) // batch_size)):
            print("Epoch [{0:2d}/{1:2d}], Step [{2:3d}/{3:3d}], Loss: {4:4f}, Elapsed time: {5:4d} seconds" \
                  .format(epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, \
                          loss.data.item(), int(time.time() - start_time)))

    # Evaluate a portion of the test data
    total, correct = get_accuracy(5000)
    print("Epoch [{0:2d}/{1:2d}], Val accuracy: {2:0.1f}% Epoch Loss: {3:4f}, Elapsed time: {4:4d} seconds" \
          .format(epoch + 1, num_epochs, float(100 * correct) / total, \
                  epoch_loss, int(time.time() - start_time)))

    # Add to the history
    run_history.append((epoch+1, epoch_loss,  float(correct) / total, int(time.time() - start_time)))

end_time = time.time()       

# =============================================================================
# Print results summary
# =============================================================================
#%%
total, correct = get_accuracy()
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
    rfile.write('Image size: {0}\n'.format(IMAGE_SIZE))
    rfile.write("Total network weights + biases: {0}\n".format(total_net_parms))
    rfile.write("Epochs: {0}\n".format(num_epochs))
    rfile.write("Learning rate: {0}\n".format(learning_rate))
    rfile.write("Batch Size: {0}\n".format(batch_size))
    rfile.write("Final loss: {0:0.4f}\n".format(loss.data.item()))
    rfile.write("Run device: {0}\n".format(run_device))
    rfile.write("Num Conv outputs: {0}\n".format(net.num_conv_outputs))
    rfile.write("Loss function: {0}\n".format(str(criterion)))
    rfile.write("Optimizer:\n{0}\n".format(str(optimizer)))
    rfile.write("Elapsed time: {0:d}\n".format(int(end_time - start_time)))
    rfile.write('\n')
    rfile.write('Accuracy of the network on the {0:d} test images: {1:0.1f}%\n'.format(total, float(100 * correct) / total))
    rfile.write('\n')
    
    for i in range(num_classes):
        rfile.write('Accuracy of {0:3s} : {1:0.1f}%\n'.format(str(i), float(100 * class_correct[i]) / class_total[i]))

with open(run_base + suffix + '_measures.csv', 'w') as mfile:
    mwriter = csv.writer(mfile)
    for obs in run_history:
        mwriter.writerow(obs)
