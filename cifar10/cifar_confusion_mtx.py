#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 10)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 10)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
TEST_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = 10
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class RelationNetwork(nn.Module):
    """RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64*2,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.fc1 = nn.Linear(input_size*2*2,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))  # torch.nn.functional.sigmoid is deprecated
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CLASS_NUM*SAMPLE_NUM_PER_CLASS,
                                            shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CLASS_NUM*TEST_NUM_PER_CLASS,
                                            shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Step 1: init data folders
    print("init data folders")

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)


    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    filename_encoder = f"./models/cifar10_feature_encoder_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl"
    filename_network = f"./models/cifar10_relation_network_{CLASS_NUM}way_{SAMPLE_NUM_PER_CLASS}shot.pkl"

    if os.path.exists(filename_encoder):
        feature_encoder.load_state_dict(torch.load(filename_encoder))
        print("load feature encoder success")
    if os.path.exists(filename_network):
        relation_network.load_state_dict(torch.load(filename_network))
        print("load relation network success")

    y_true = []
    y_pred = []
    for _ in range(TEST_EPISODE):

        # kmanakk1 - change how we get samples for compatability with pytorch 1.7
        sample_iterator = iter(trainloader)
        sample_images,sample_labels = next(sample_iterator)

        for test_images,test_labels in testloader:
            batch_size = test_labels.shape[0]
            # calculate features
            sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,8,8)
            sample_features = torch.sum(sample_features,1).squeeze(1)
            test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

            # calculate relations
            # each batch sample link to every samples to calculate relations
            # to form a 100x128 matrix for relation network
            sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

            test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
            test_features_ext = torch.transpose(test_features_ext,0,1)
            relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,8,8)
            relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

            _,predict_labels = torch.max(relations.data,1)

            # save outputs and truth
            y_true.extend(test_labels.to("cpu"))
            y_pred.extend(predict_labels.to("cpu"))

    # make confusion matrix
    cf_mtx = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cf_mtx / np.sum(cf_mtx, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))
    sn.heatmap(df, annot=True)
    if not os.path.exists('images'): os.makedirs('images')
    plt.savefig('images/cifar_confusion_mtx.png')
if __name__ == '__main__':
    main()
