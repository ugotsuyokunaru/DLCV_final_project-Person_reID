"""
  FileName     [ train.py ]
  PackageName  [ final ]
  Synopsis     [ Train the Person_reID model ]
  Dataset:
  - IMDb
  Dataloader: Customized Image Loader
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
  - python3 train.py --name ft_net_dense
"""






from __future__ import division, print_function

import argparse
# import math
import os
import time
from functools import reduce
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

import evaluate_rerank
import evaluate_gpu
import utils
from imdb_master import IMDbTrainset
from model_res50 import FeatureExtractorFace, Classifier
from pcb_extractor import extract_feature
from random_erasing import RandomErasing

matplotlib.use('agg')

# -----------------------------------------
# fp16: Use Float16 to train the network.
# -----------------------------------------
# try:
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
# except ImportError: # will be 3.x series
#     print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

######################################################################
parser = argparse.ArgumentParser(description='Training')
# Model Setting
# parser.add_argument('--num_part', default=6, type=int, help='A parameter of PCB network.')
# parser.add_argument('--stride', default=2, type=int, help='stride')
# parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
# parser.add_argument('--use_NAS', action='store_true', help='use NAS' )
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )

parser.add_argument('--keep_others', default=False, help='if true, the image of type others will be keeped.')
parser.add_argument('--cast_image', default=True, help='if true, the cast image will train with candidates.')
# parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
# parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
# parser.add_argument('--img_size', default=[448, 448], type=int, nargs='*')
# Training setting
parser.add_argument('--batchsize', default=64, type=int, help='batchsize in training')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--milestones', default=[10, 20, 30, 40], nargs='*', type=int)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--epochs', default=50, type=int)
# parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
# parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
# parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
# parser.add_argument('--optimizer', default='SGD', type=str, help='choose optimizer')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--b1', default=0.9, type=float)
parser.add_argument('--b2', default=0.999, type=float)
# I/O Setting 
parser.add_argument('--mpath', default='CE', type=str, help='output model name')
parser.add_argument('--resume', type=str, help='If true, resume training at the checkpoint')
parser.add_argument('--trainset', default='IMDb_resize/train', type=str, help='Directory of training set.')
parser.add_argument('--valset', default='IMDb_resize/val', type=str, help='Directory of validation set')
# Device Setting
parser.add_argument('--gpu_ids', default=0, nargs='*', type=int, help='')
parser.add_argument('--threads', default=0, type=int)
# Others Setting
parser.add_argument('--debug', action='store_true', help='use debug mode (print shape)' )
parser.add_argument('--log_interval', default=10, type=int)
parser.add_argument('--save_interval', default=2, type=int)

opt = parser.parse_args()


def val(feature_extractor, classifier, loader, epoch, device):    
    

    feature_extractor.eval()
    classifier.eval()
    
    feature_extractor.to(device)
    classifier.to(device)

    # -------------------------------- #
    # Extract the features             #
    # -------------------------------- #
    
 
    features = extract_feature(test_model, loader)
    num_candidates = loader.dataset.candidates.shape[0]    
    candidate_feature = features[:num_candidates]
    cast_feature = features[num_candidates:]

    print("Extracted_features.shape: {}".format(features.shape))

    candidate_paths = loader.dataset.candidates['level_1']
    candidate_films = loader.dataset.candidates['level_0']
    cast_paths = loader.dataset.casts['level_1']
    cast_films = loader.dataset.casts['level_0']

    model.cuda()

    # -------------------------------- # 
    # Calculate the mAP # 
    # -------------------------------- #

    # re_rank = evaluate_rerank.run(cast_feature, candidate_feature, opt.k1, opt.k2, opt.lambda_value)
    # print(re_rank)

    mAP = evaluate_gpu.run(cast_feature, cast_names, cast_films, candidate_feature, candidate_names, candidate_films, opt.valset + "_GT.json", "result.csv")
 
    
    return mAP

def train(feature_extractor, classifier, criterion, optimizer, scheduler,device,  num_epochs=25, save_freq=1, debug=False):
    # Warm starting training technique.
    # warm_up = 0.1 # We start from the 0.1 * lrRate
    # warm_iteration = round(len(dataloaders['train'].dataset) / opt.batchsize) * opt.warm_epoch # first 5 epoch

    if debug:
        model.debug_mode()
        
    # for epoch in range(1, num_epochs + 1):

    scheduler.step()
    
    feature_extractor.train()
    classifier.train()
    
    feature_extractor.to(device)
    classifier.to(device)
    
    running_loss = 0.0
    running_corrects = 0.0

    # Iterate over data.
    for index, (inputs, labels) in enumerate(dataloaders['train'], 1):
        n = inputs.shape[0]
        optimizer.zero_grad()                
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # print("Inputs.shape: ", inputs.shape)
        # print("Labels.shape: ", labels.shape)

        outputs = feature_extractor(inputs)
        outputs = classifier(outputs)

        if not opt.PCB:
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
        # backward + optimize only if in training phase
        # if epoch < opt.warm_epoch: 
        #     warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
        #     loss *= warm_up

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * n
        running_corrects += float(torch.sum(preds == labels.data))
        
        # Temporal training informations
        corrects = torch.mean((preds == labels.data).type(torch.float))
        if index % opt.log_interval == 0:
            print('[Train] [Epoch {:2d}/{:2d}] [Iteration {:4d}/{:4d}] [Loss: {:.4f}] [Running Acc: {:.2%}]'.format(
                epoch, num_epochs, index, len(dataloaders['train']), loss.item(), running_corrects / n / index))
        
    epoch_loss = running_loss / len(image_datasets['train'])
    epoch_acc  = running_corrects / len(image_datasets['train'])

    y['train_loss'].append(epoch_loss)
    y['train_acc'].append(epoch_acc)
    
    return feature_extractor, classifier, epoch_loss, epoch_acc


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
plt.figure(figsize=(19.2, 10.8))
'''
def draw_curve(x, y, save_jpg='train.jpg'):
    plt.clf()

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(x, y['train_loss'], 'bo-', label='Train Loss')
    plt.legend(loc=0)

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(x, y['train_acc'], 'go-', label='Train Acc')
    plt.plot(x, y['val_mAP'], 'ro-', label='Validation mAP')
    plt.legend(loc=0)

    plt.savefig(os.path.join('./model', opt.name, save_jpg))


'''
######################################################################
# Save model
#---------------------------
def save_network(network, epoch, device, opt, num_fill=3):
    # os.makedirs(opt.mpath, exist_ok=True)
    save_path = os.path.join('models',opt.mpath,
			 'net_{}.pth'.format(str(epoch).zfill(num_fill)))
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.to(device)

    return

# ------------------------------ # 
# Finetuning the convolution-net #
# ------------------------------ #
#
# Load a pretrainied model and reset final fully connected layer.
#

# set gpu ids
#if len(opt.gpu_ids) > 0:
#    torch.cuda.set_device(opt.gpu_ids[0])
#    cudnn.benchmark = True

#opt.img_size = tuple(opt.img_size)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_ids)
device = torch.device("cuda")

# ---------------------------------
# Dataaugmentation setting
# ---------------------------------
transform_train_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

transform_val_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets['train'] = IMDbTrainset(
    movie_path=opt.trainset, 
    feature_path=None, 
    label_path=opt.trainset+"_GT.json",
    keep_others=opt.keep_others,
    cast_image=opt.cast_image,
    mode='classify',
    debug=opt.debug, 
    transform=data_transforms['train']
)
image_datasets['val'] = IMDbTrainset(
    movie_path=opt.valset, 
    feature_path=None, 
    label_path=opt.valset+"_GT.json",
    keep_others=True,
    cast_image=True,
    mode='features',
    debug=opt.debug, 
    transform=data_transforms['val']
)

dataloaders = {}
# pin_memory = True for good GPU (ref : https://blog.csdn.net/tsq292978891/article/details/80454568 )
dataloaders['train'] = torch.utils.data.DataLoader(
    image_datasets['train'], 
    batch_size=opt.batchsize, 
    drop_last=True,
    shuffle=True, 
    num_workers=opt.threads,
    pin_memory=True
)
dataloaders['val'] = torch.utils.data.DataLoader(
    image_datasets['val'], 
    batch_size=opt.batchsize,
    shuffle=True, 
    num_workers=opt.threads, 
    pin_memory=True
)

class_names = image_datasets['train'].classes
opt.nclasses  = len(class_names)
opt.len_class = len(class_names) 
# For saving in yaml
# print(class_names)

#use_gpu = torch.cuda.is_available()
# DEVICE = utils.selectDevice()


#---------------------------------
# load face pretrained model
#---------------------------------

feature_extractor = FeatureExtractorFace().to(device)
classifier = Classifier(fc_out=opt.len_class).to(device)


print(feature_extractor)
print(classifier)

optimizer = torch.optim.Adam(
                classifier.parameters(), 
                lr=opt.lr,
                weight_decay=opt.weight_decay,
                betas=(opt.b1, opt.b2)
            )
  
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    
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

y = {
    'train_loss': [],
    'train_acc': [],
    'val_mAP': []
}





# ---------------------------------------- # 
# Optimizer for different model Structures # 
# ---------------------------------------- #



if __name__ == "__main__":
    
    dir_name = os.path.join('./models', opt.mpath)
    os.makedirs(dir_name, exist_ok=True)

    # record every run
#    copyfile('./train.py', os.path.join(dir_name, 'train.py'))
#    copyfile('./model.py', os.path.join(dir_name, 'model.py'))

    # save opts
#    with open('./{}/opts.yaml'.format(dir_name),'w') as fp:
#        yaml.dump(vars(opt), fp, default_flow_style=False)

    # print opts
    utils.details(opt)

    # ---------------- # 
    # training setting # 
    # ---------------- # 

    criterion = nn.CrossEntropyLoss()

    since = time.time()

    # ------------------ # 
    # Train & Validation # 
    # ------------------ # 
    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    # best_mAP = 0.0

    # with torch.no_grad():
    #     val_mAP = val(model, dataloaders['val'], 0)

    for epoch in range(1, opt.epochs + 1):
        # Train
        feature_extractor, classifier, epoch_loss, epoch_acc = train(
                feature_extractor,classifier,
                criterion, optimizer, scheduler, device, 
                num_epochs=opt.epochs, save_freq=opt.save_interval, debug=opt.debug
                )

        # Save
        if epoch % opt.save_interval == 0:
            save_network(classifier, epoch, device, opt)

        # Validation
        with torch.no_grad():
            val_mAP = val(feature_extractor, classifier, dataloaders['val'], epoch, device)
        
        y['val_mAP'].append(val_mAP)

        # best_model_wts = model.state_dict()
        # best_mAP = mAP
        
        # Draw curves
        x_epoch.append(epoch)
#        draw_curve(x_epoch, y)
        
        # Print times
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
  
      
    # ------------------ # 
    # End                # 
    # ------------------ # 
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
