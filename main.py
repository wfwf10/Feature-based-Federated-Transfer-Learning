import torch
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime

#######################################
### PRE-TRAINED MODELS AVAILABLE HERE
## https://pytorch.org/docs/stable/torchvision/models.html
from torchvision import models
#######################################

now = datetime.datetime.now()

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

##########################
### SETTINGS
##########################

FL_type = 'FbFTL' 
train_set_denominator = 'full'  # 'full', int <= 50000  # pick a subset with 50000/int training samples

# Hyperparameters
NUM_CLASSES = 10
random_seed = 1 
learning_rate = 1e-2  
num_epochs = 200  
batch_size = 64  
momentum = 0.9  
lr_decay = 5e-4  

write_hist = True

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = learning_rate * (0.5 ** ((epoch * 10) // num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_seed) # Sets the seed for generating random numbers for the current GPU. 
torch.manual_seed(random_seed) # sets the seed for generating random numbers.

if write_hist:
    file1 = open('history.txt', 'a')
    file1.write('\n \n \n Time:')
    file1.write(str(now.year) + ' ' + str(now.month) + ' ' + str(now.day) + ' ' 
                + str(now.hour) + ' ' + str(now.minute) + ' ' + str(now.second) 
                + '     ' + FL_type + ',  train_deno:' + str(train_set_denominator))
    file1.close()

##########################
### CIFAR10 DATASET
##########################

custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='data', train=True, transform=custom_transform,download=True)
test_dataset = datasets.CIFAR10(root='data', train=False, transform=custom_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

if train_set_denominator == 'full':
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
else:
    selected_list = list(range(0, len(train_dataset), train_set_denominator))
    trainset_1 = torch.utils.data.Subset(train_dataset, selected_list)
    train_loader = torch.utils.data.DataLoader(dataset=trainset_1, batch_size=batch_size, num_workers=8, shuffle=True)

##########################
### LOAD MODEL
##########################

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier[3].requires_grad = True
model.classifier[6] = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))

##########################
### TRAIN MODEL
##########################

model = model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lr_decay, momentum=momentum)

def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(data_loader):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss
    
    

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    accuracy = compute_accuracy(model, test_loader)
    loss = compute_epoch_loss(model, test_loader)
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % (epoch+1, num_epochs, accuracy, loss))
        if write_hist:
            file1 = open('history.txt', 'a')
            file1.write('\n Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % (epoch+1, num_epochs, accuracy, loss))
            file1.close()


    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    accuracy = compute_accuracy(model, test_loader)
    print('Test accuracy: %.2f%%' % (accuracy))
    if write_hist:
        file1 = open('history.txt', 'a')
        file1.write('\n Test accuracy: %.2f%%' % (accuracy))
        file1.close()


