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
import cv2
from PIL import Image
from CNN4Module import CNNModule
from myresnet18 import ResNet18
import tensorwatch as tw

model = torch.load('norm_cnn4.pkl')
tw.draw_model(model,[1,1,28,28])