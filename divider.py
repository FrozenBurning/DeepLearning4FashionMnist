
import pandas as pd
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from sklearn.metrics import accuracy_score,f1_score,roc_curve,precision_recall_curve,average_precision_score,auc
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix,matthews_corrcoef,roc_auc_score
import matplotlib.pyplot as plt
import torch.utils.data as Data
import numpy as npy
import math
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io,transform
import cv2
import random
from PIL import Image


raw_train_data = npy.load('train.npy')
raw_train_label = pd.read_csv('train.csv')
raw_train_label = (raw_train_label.values)[:,1] # to numpy
raw_train_label = raw_train_label.astype('int64')

validation_split = .1
shuffle_dataset = True
random_seed= random.randint(0,100)
# Creating data indices for training and validation splits:
dataset_size = len(raw_train_data)
indices = list(range(dataset_size))
split = int(npy.floor(validation_split * dataset_size))
if shuffle_dataset :
    npy.random.seed(random_seed)
    npy.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


train_set = npy.zeros((len(train_indices),784))
valid_set = npy.zeros((len(val_indices),784))
train_label_set = (npy.zeros(len(train_indices))).astype('int64')
valid_label_set = (npy.zeros(len(val_indices))).astype('int64')

des = 0
for i in train_indices:
    train_set[des,:] = raw_train_data[i,:]
    train_label_set[des] = raw_train_label[i]
    des +=1

des = 0
for i in val_indices:
    valid_set[des,:] = raw_train_data[i,:]
    valid_label_set[des] = raw_train_label[i]
    des += 1

npy.save('train_set.npy',train_set)
npy.save('train_set_label.npy',train_label_set)
npy.save('valid_set.npy',valid_set)
npy.save('valid_set_label.npy',valid_label_set) 

