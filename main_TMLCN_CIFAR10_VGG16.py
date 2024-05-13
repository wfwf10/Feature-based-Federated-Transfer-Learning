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
import math
from collections import deque
import copy

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

FL_type = 'FbFTL'  # 'FL', 'FTLf', 'FTLc', 'FbFTL'
train_set_denominator = 'full'  # 'full', int <= 50000  # pick a subset with 50000/int training samples

# Hyperparameters
NUM_CLASSES = 10
U_clients = 6250  # number of clients, 50000/8
random_seed = 1  # transfer = True / False
learning_rate = 1e-2  # 1e-3, 0.05, 1e-2
num_epochs = 200  # 10, 300, 200
batch_size = 64  # 128, 128, (out of memory:64)
momentum = 0.9  # None, 0.9
lr_decay = 5e-4  # 1e-6, 5e-4
if FL_type == 'FL':
    transfer, full = False, True  # transfer or train model from scratch   # train whole model or last few layers
    sigma = 0.  # 0.5 relative std for addtive gaussian noise on gradients
elif FL_type == 'FTLf':
    transfer, full = True, True  
    sigma = 0.  # 0.3305 relative std for addtive gaussian noise on gradients
elif FL_type == 'FTLc':
    transfer, full = True, False 
    sigma = 0.  # 0.285 relative std for addtive gaussian noise on gradients
elif FL_type == 'FbFTL':
    transfer, full = True, False 
    sigma = 0  # 0.8? relative std for addtive gaussian noise on features
    saved_noise = True  # save noise at beginning
else:
    raise ValueError('Unknown FL_type: ' + FL_type)
relative_noise_type = 'all_std'  # 'individual', 'all_std'
packet_loss_rate = 0.  # 0, 0.05, 0.1, 0.15
quan_digit = 32  # digits kept after feature quantization: None (max:(12~18)(6~8), min=0, std~0.8) or int
sparse_rate = 0.9  # ratio of uplink elements kept after sparsification: None or (0,1]
class ErrorFeedback(object):
    queue = deque(maxlen=U_clients)
    temp = deque()
if (quan_digit or sparse_rate) and FL_type != 'FbFTL':
    errfdbk = ErrorFeedback()
    # print(errfdbk.queue, errfdbk.temp)

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
    file1 = open('/data1/feng/LaTFL/history.txt', 'a')
    file1.write('\n \n \n Time:')
    file1.write(str(now.year) + ' ' + str(now.month) + ' ' + str(now.day) + ' ' + str(now.hour) + ' ' 
                + str(now.minute) + ' ' + str(now.second) + '     ' + FL_type 
                # + ',  train_deno:' + str(train_set_denominator)
                # + ',  sigma:' + str(sigma)
                # + ',  packet_loss_rate:' + str(packet_loss_rate)
                + ',  quantization digits:' + str(quan_digit)
                + ',  sparsification rate:' + str(sparse_rate)
                )
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

## Note that this particular normalization scheme is necessary since it was used for pre-training the network on ImageNet.
## These are the channel-means and standard deviations for z-score normalization.

train_dataset = datasets.CIFAR10(root='data', train=True, transform=custom_transform,download=True)
test_dataset = datasets.CIFAR10(root='data', train=False, transform=custom_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

if train_set_denominator == 'full':
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    train_set_len = len(train_dataset)
else:
    # print('len(train_dataset)', len(train_dataset))  # 50000
    # selected_list = list(range(0, len(train_dataset), 2))
    selected_list = list(range(0, len(train_dataset), train_set_denominator))
    trainset_1 = torch.utils.data.Subset(train_dataset, selected_list)
    train_loader = torch.utils.data.DataLoader(dataset=trainset_1, batch_size=batch_size, num_workers=8, shuffle=True)
    train_set_len = len(selected_list)

# # Checking the dataset
# for images, labels in train_loader:  
#     print('Image batch dimensions:', images.shape)
#     print('Image label dimensions:', labels.shape)
#     break

# labels = torch.zeros(10, dtype=torch.long)
# for batch_idx, (features, targets) in enumerate(train_loader):
#     for t in targets:
#         labels[t] += 1
# print('labels', labels)

##########################
### LOAD MODEL
##########################

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """
    def __init__(self, sigma=0, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        if saved_noise:
            self.register_buffer('noise', torch.empty(train_set_len*4096).normal_(mean=0,std=1))
            self.i = 0
        else:
            self.register_buffer('noise', torch.tensor(0))

    def forward(self, x):
        if self.training and quan_digit:
            # print(x)
            # print(x.dtype, torch.max(x), torch.min(x), torch.std(x))
            x = torch.round((2**quan_digit-1) / torch.max(x) * x) * torch.max(x) / (2**quan_digit-1)
            # print(x.dtype, torch.max(x), torch.min(x), torch.std(x))
            # print(x)
            # quit()
        if self.training and self.sigma != 0:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.cuda.manual_seed(random_seed) # Sets the seed for generating random numbers for the current GPU. 
            torch.manual_seed(random_seed) # sets the seed for generating random numbers.
            if relative_noise_type == 'individual':
                scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
                sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            elif relative_noise_type == 'all_std':
                x_std = torch.std(x.detach()) if self.is_relative_detach else torch.std(x)
                # print(*x.size())  # 64 4096
                if saved_noise:
                    sampled_noise = torch.reshape(self.noise[self.i*batch_size*4096 : (self.i+1)*batch_size*4096],(-1, 4096)
                                                  ).detach().float() * x_std * self.sigma
                    self.i = self.i + 1 if (self.i+1)*batch_size*4096<train_set_len*4096 else 0
                else:
                    sampled_noise = self.noise.expand(*x.size()).float().normal_(std=x_std*self.sigma)
            x = x + sampled_noise
        return x 
        
    def set_sigma(self, sigma):
        self.sigma = sigma

if not full:
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[3].requires_grad = True
    # model.classifier[6] = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    if FL_type == 'FbFTL':
        model.classifier[6] = nn.Sequential(GaussianNoise(sigma=sigma), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), 
                                            nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    else:
        model.classifier[6] = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), 
                                            nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
else:  # full
    if transfer:
        model = models.vgg16(pretrained=True)
    if not transfer:
        # model = models.vgg16(pretrained=False)
        model = models.vgg16(pretrained=True)
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.Conv2d):
                # print('m', m)
                # print(m.in_channels, m.out_channels)
                # print(m.__class__.__name__)
                # if m.out_channels > 256:
                m.reset_parameters()
        for i in range(21, 31):
            model.features[i].apply(weight_reset)
        for i in range(7):
            model.classifier[i].apply(weight_reset)

        for param in model.parameters():
            param.requires_grad = True
    
    # model.classifier[3].requires_grad = True  # (4096, 4096, relu, dropout(0.5))
    # model.classifier[6] = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))
    model.classifier[6] = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 512), 
                                        nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, NUM_CLASSES))


def Gaussian_noise_to_weights(m):
    if sigma!=0:
        with torch.no_grad():
            for param in m.parameters():
                if param.requires_grad:
                    if relative_noise_type == 'individual':
                        # print(param.grad.view(-1))
                        # print(param)
                        # scale = sigma * param.detach()
                        scale = sigma * param.grad.detach()  # todo: * math.sqrt(batch_size/8)
                        noise = torch.tensor(0).to(DEVICE)
                        sampled_noise = noise.expand(*param.size()).float().normal_() * scale
                    elif relative_noise_type == 'all_std':
                        param_grad_std = torch.std(param.grad.detach()) 
                        noise = torch.tensor(0).to(DEVICE)
                        sampled_noise = noise.expand(*param.size()).float().normal_(std=param_grad_std*sigma)  # todo: * math.sqrt(batch_size/8)
                    # param = param + sampled_noise
                    param.add_(sampled_noise)

def Errfdbk_to_weights(m):
    print("inner model.apply")
    with torch.no_grad():
        for param in m.parameters():
            print('len(param)', len(param))
            if param.requires_grad:
                print('len(errfdbk.queue)', len(errfdbk.queue))
                # print(errfdbk.temp)
                print('len(errfdbk.temp)', len(errfdbk.temp))
                p_grad = param.grad.detach()
                if err_flag:
                    p_grad += errfdbk.temp.popleft()
                p_grad_qs = copy.deepcopy(p_grad)
                if sparse_rate:
                    pass
                if quan_digit:
                    pass
                err = p_grad_qs - p_grad
                param.add_(err)
                errfdbk.temp.append(-err)
    print('seems good')

if FL_type == 'FbFTL':
    received_batches_FbFTL = np.ones(len(train_loader))
    received_batches_FbFTL[:int(len(train_loader)*packet_loss_rate)] = 0
    np.random.shuffle(received_batches_FbFTL)
    
def Packet_Received(batch_idx):
    if FL_type == 'FbFTL':
        return received_batches_FbFTL[batch_idx]
    else:
        return np.random.choice(2, p=[packet_loss_rate, 1-packet_loss_rate]) 

# for a in range(3):
#     print('round', a)
#     for i in range(50):
#         print(Packet_Received(i))

# print(model)

# for name, param in model.named_parameters():
#     print(name, torch.numel(param), param.requires_grad)
# quit()

# model(torch.randn(1, 3, 224, 224)).mean().backward()
# for name, param in model.named_parameters():
#     print(name, param.grad)
#     print('value', param.data)
#     wait = input('next layer')

##########################
### TRAIN MODEL
##########################

model = model.to(DEVICE)
# if transfer:
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# else:
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lr_decay, momentum=momentum)  # , nesterov=True)
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
    # adjust_learning_rate(optimizer, epoch)
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        if Packet_Received(batch_idx):
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### PRIVACY NOISE
            if FL_type != 'FbFTL':
                # Gaussian_noise_to_weights(model, sigma * math.sqrt(batch_size/8))
                model.apply(Gaussian_noise_to_weights)

            ### Sparsification/Quantization with Error Feedback  # errfdbk.queue = deque(maxlen=U_clients)
            if (quan_digit or sparse_rate) and FL_type != 'FbFTL':
                print('main loop: len(errfdbk.queue)', len(errfdbk.queue))
                if len(errfdbk.queue) < U_clients:
                    errfdbk.temp = deque()
                    err_flag = False
                else:
                    errfdbk.temp = errfdbk.queue.popleft()
                    err_flag = True
                model.apply(Errfdbk_to_weights)
                print('completed one cycle!')
                errfdbk.queue.append(errfdbk.temp)
        
            ### LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % (epoch+1, num_epochs, batch_idx, len(train_loader), cost))
                # if FL_type == 'FbFTL':
                    # print(model.classifier[6][0].sigma)
                    # model.classifier[6][0].set_sigma(sigma=0.4) 

    model.eval()
    accuracy = compute_accuracy(model, test_loader)
    loss = compute_epoch_loss(model, test_loader)
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % (epoch+1, num_epochs, accuracy, loss))
        if write_hist:
            file1 = open('/data1/feng/LaTFL/history.txt', 'a')
            file1.write('\n Epoch: %03d/%03d | Test: %.3f%% | Loss: %.3f' % (epoch+1, num_epochs, accuracy, loss))
            file1.close()


    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

with torch.set_grad_enabled(False): # save memory during inference
    accuracy = compute_accuracy(model, test_loader)
    print('Test accuracy: %.2f%%' % (accuracy))
    if write_hist:
        file1 = open('/data1/feng/LaTFL/history.txt', 'a')
        file1.write('\n Test accuracy: %.2f%%' % (accuracy))
        file1.close()

# model.save_weights('/data1/feng/LaTFL/cifar10vgg.h5')

