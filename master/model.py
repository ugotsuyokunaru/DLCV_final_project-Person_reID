"""
  FileName     [ model.py ]
  PackageName  [ layumi/Person_reID_baseline_pytorch ]
  Synopsis     [ Model class ]

  Library:
  - apex: A PyTorch Extension, Tools for easy mixed precision and distributed training in Pytorch
          https://github.com/NVIDIA/apex
  - yaml: A human-readable data-serialization language, and commonly used for configuration files.
  - pretrainedmodels: 
          Install: pip install pretrainedmodels

  Pretrain network:
  - PCB: Part-based Convolutional Baseline
         https://arxiv.org/abs/1711.09349
         Beyond Part Models: Person Retrieval with Refined Part Pooling (and a Strong Convolutional Baseline)
  - DenseNet:
  - NAS:
  - ResNet: 
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torchvision import models

import pretrainedmodels


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
        
    def forward(self, x):
        x = self.add_block(x)
        
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, debug=False):
        super(ft_net, self).__init__()
        self.debug = debug
        model_ft = models.resnet50(pretrained=True)

        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)

    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in ft_net : {}".format(x.shape))

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)

        if self.debug:
            print("shape before view x in ft_net : {}".format(x.shape))

        x = x.view(x.size(0), x.size(1))

        if self.debug:
            print("shape after view(classifier input) x in ft_net : {}".format(x.shape))

        x = self.classifier(x)

        if self.debug:
            print("shape after classifier x in ft_net : {}".format(x.shape))
            print()
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, debug=False):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.debug = debug
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num, droprate)

    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in ft_net_dense : {}".format(x.shape))

        x = self.model.features(x)

        if self.debug:
            print("shape before view x in ft_net_dense : {}".format(x.shape))

        x = x.view(x.size(0), x.size(1))

        if self.debug:
            print("shape after view (classifier input) x in ft_net_dense : {}".format(x.shape))

        x = self.classifier(x)

        if self.debug:
            print("shape after classifier x in ft_net_dense : {}".format(x.shape))
            print()
        return x

# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, debug=False):
        super().__init__()  
        model_name = 'nasnetalarge' 
        # pip install pretrainedmodels
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.debug = debug
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)
    
    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in ft_net_NAS : {}".format(x.shape))

        x = self.model.features(x)
        x = self.model.avg_pool(x)

        if self.debug:
            print("shape befor view x in ft_net_NAS : {}".format(x.shape))

        x = x.view(x.size(0), x.size(1))

        if self.debug:
            print("shape after view x in ft_net_NAS : {}".format(x.shape))

        x = self.classifier(x)

        if self.debug:
            print("shape after classifier x in ft_net_NAS : {}".format(x.shape))
            print()

        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5, debug=False):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.debug = debug
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in ft_net_middle : {}".format(x.shape))

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)

        if self.debug:
            print("shape befor cat view x in ft_net_middle : {}".format(x.shape))

        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))

        if self.debug:
            print("shape after cat view x in ft_net_middle : {}".format(x.shape))

        x = self.classifier(x)

        if self.debug:
            print("shape after classifier x in ft_net_middle : {}".format(x.shape))
            print()
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num, debug=False):
        super(PCB, self).__init__()

        self.debug = debug
        self.part  = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in PCB : {}".format(x.shape))

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        if self.debug:
            print("shape before 6 parts x in PCB : {}".format(x.shape))

        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        y = []
        for i in range(self.part):
            y.append(predict[i])

        if self.debug:
            print("each shape of 6 parts prediction in PCB : {}".format(y[0].shape))
            print()

        return y

class PCB_test(nn.Module):
    def __init__(self, model, debug=False):
        super(PCB_test,self).__init__()

        self.debug = debug
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def debug_mode(self):
        self.debug = True

    def forward(self, x):
        if self.debug:
            print("shape of input x in PCB_test : {}".format(x.shape))
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)

        if self.debug:
            print("shape before view x in PCB_test : {}".format(x.shape))

        y = x.view(x.size(0), x.size(1), x.size(2))
        
        if self.debug:
            print("shape of output (after view) x in PCB_test : {}".format(x.shape))
            print()

        return y

def model_structure_unittest():
    """ Debug model structure """
    net = ft_net(751, stride=1)
    net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    print(output.shape)

if __name__ == "__main__":
    model_structure_unittest()
