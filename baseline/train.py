"""
  FileName     [ train.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Train the Person_reID model ]

  Dataset:
  - Market1501

  Dataloader: Default Image Loader

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.
  - shutil: High-level file operations Library

  Pretrain network:
  - PCB:
  - DenseNet:
  - NAS:
  - ResNet: 

  Usage:
  - python3 train.py --name PCB --PCB --lr 0.02 --batchsize 16 --debug
  - python3 train.py --name PCB --PCB --lr 0.02 --batchsize 16
"""

from __future__ import division, print_function

import argparse
import math
import os
#from PIL import Image
import time
from functools import reduce
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

import utils
import yaml
from model import PCB, ft_net, ft_net_dense, ft_net_NAS
from random_erasing import RandomErasing

matplotlib.use('agg')

# -----------------------------------------
# fp16: Use Float16 to train the network.
# -----------------------------------------
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default=[0], nargs='*', type=int, help='')
parser.add_argument('--name', default='ft_net_dense', type=str, help='output model name')
parser.add_argument('--trainset', default='./Market/pytorch/train', type=str, help='Directory of training set.')
parser.add_argument('--valset', default='./Market/pytorch/val', type=str, help='Directory of validation set')
parser.add_argument('--num_part', default=6, type=int, help='A parameter of PCB network.')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
opt = parser.parse_args()

fp16 = opt.fp16
name = opt.name

# set gpu ids
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])
    cudnn.benchmark = True

# ---------------------------------
# Dataaugmentation setting
# ---------------------------------
transform_train_list = [
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256,128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(opt.trainset, data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(opt.valset, data_transforms['val'])

# pin_memory = True for good GPU (ref : https://blog.csdn.net/tsq292978891/article/details/80454568 )
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, drop_last=True,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes
print(class_names)


# train / val class_names check:
class_names_val = image_datasets['val'].classes
count = 0
for val_cls in class_names_val:
    if val_cls in class_names:
        count += 1
print("\ntrain_class : ", len(class_names))
print("val_class : ", len(class_names_val))
print("same class :", count, '\n')

use_gpu = torch.cuda.is_available()
DEVICE = utils.selectDevice()

# Show I/O time delay for 1 iterations 
# since = time.time()
inputs_train, classes_train = next(iter(dataloaders['train']))
inputs_val, classes_val = next(iter(dataloaders['val']))
print('inputs_train :', inputs_train.shape)
print('classes_train :', classes_train.shape, '\n',  classes_train, '\n')
print('inputs_val :', inputs_val.shape)
print('classes_val :', classes_val.shape, '\n', classes_val, '\n')
print()
# print(time.time() - since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, save_freq=10, debug=False):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    # Warm starting training technique.
    warm_up = 0.1 # We start from the 0.1*lrRate
    warm_iteration = round(len(dataloaders['train'].dataset) / opt.batchsize) * opt.warm_epoch # first 5 epoch

    if debug:
        model.debug_mode()
    for epoch in range(1, num_epochs + 1):
        scheduler.step()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':    
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            
            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase], 1):
                now_batch_size, c, h, w = inputs.shape
                optimizer.zero_grad()                

                # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # print(inputs.shape)
                # print(labels.shape)

                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    if debug:
                        print("outputs :", outputs)
                        print("labels :", labels, '\n')
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    for i in range(opt.num_part):
                        part[i] = outputs[i]

                    # score = reduce((lambda x, y: x + y), [F.softmax(tensor, dim=1) for tensor in part.values()])
                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    if debug:
                        print("part[0] : ", part[0])
                        print("labels : ", labels , '\n')

                    loss = criterion(part[0], labels)
                    for i in range(opt.num_part - 1):
                        loss += criterion(part[i+1], labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    optimizer.step()

                # statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

                if debug:
                    print("preds :", preds)
                    print("labels.data :", labels.data, '\n')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('[{}] [Epoch {}/{}] [Loss: {:.4f}] [Acc: {:.2%}]'.format(phase, epoch, num_epochs, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)            

            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()

                if epoch % save_freq == 0:
                    save_network(model, epoch)

                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    # ------------------------
    # All training epochs ends
    # ------------------------
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(current_epoch, save_jpg='train.jpg'):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')

    if current_epoch == 0:
        ax0.legend()
        ax1.legend()

    fig.savefig(os.path.join('./model', name, save_jpg))

######################################################################
# Save model
#---------------------------
def save_network(network, epoch):
    num_fill = 3
    save_path = os.path.join('./model', name, 'net_{}.pth'.format(str(epoch).zfill(num_fill)))
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.cuda(opt.gpu_ids[0])

######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense(len(class_names), opt.droprate)
elif opt.use_NAS:
    model = ft_net_NAS(len(class_names), opt.droprate)
else:
    model = ft_net(len(class_names), opt.droprate, opt.stride)

if opt.PCB:
    model = PCB(len(class_names))

opt.nclasses = len(class_names)

print(model)

# ---------------------------------------- # 
# Optimizer for different model Structures # 
# ---------------------------------------- #
if not opt.PCB:
    ignored_params = list(map(id, model.classifier.parameters() ))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters() ))
    ignored_params += (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                     +list(map(id, model.classifier4.parameters() ))
                     +list(map(id, model.classifier5.parameters() ))
                     #+list(map(id, model.classifier6.parameters() ))
                     #+list(map(id, model.classifier7.parameters() ))
                      )
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.model.fc.parameters(), 'lr': opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
             {'params': model.classifier4.parameters(), 'lr': opt.lr},
             {'params': model.classifier5.parameters(), 'lr': opt.lr},
             #{'params': model.classifier6.parameters(), 'lr': 0.01},
             #{'params': model.classifier7.parameters(), 'lr': 0.01}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#

if __name__ == "__main__":
    dir_name = os.path.join('./model', name)
    os.makedirs(dir_name,mode=0o777, exist_ok=True)
    # if not os.path.isdir(dir_name):
    #     os.mkdir(dir_name)

    # record every run
    copyfile('./train.py', os.path.join(dir_name, 'train.py'))
    copyfile('./model.py', os.path.join(dir_name, 'model.py'))

    # save opts
    with open('./{}/opts.yaml'.format(dir_name),'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

    # model to gpu
    # model = model.to(DEVICE)
    model = model.cuda()
    if fp16:
        # model = network_to_half(model)
        # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")

    criterion = nn.CrossEntropyLoss()
    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=60, save_freq=2, debug=opt.debug)
