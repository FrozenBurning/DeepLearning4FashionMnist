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

img_size = 28

class MyDataset(Data.Dataset):
    def __init__(self, images, labels,transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):#返回的是tensor
        tmp_mat = (self.images[index]).reshape(img_size,img_size)
        img,target =tmp_mat,[]
        img = img.astype(npy.float32)
        img = img/255.0
        # print(img)
        # print("label: "+str(target))
        if self.transform:
            tmp = Image.fromarray(img)
            img = self.transform(tmp)
        else:
            img = img[npy.newaxis,:]
        return img, target

    def __len__(self):
        return len(self.images)

batch_size = 64

useResnet = False
useOfficialData = True
ModelFusion = True

if ModelFusion:
    # model1 = torch.load('norm_60resmodel.pkl')
    model1 = torch.load('norm_incepmodel.pkl')
    # model2 = torch.load('89533cnn4model.pkl')
    # model2 = torch.load('norm_res18.pkl')
    model2 = torch.load('norm_cnn4.pkl')
    model3 = torch.load('norm_res18.pkl')
    model4 = torch.load('norm_60resmodel.pkl')
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
else:
    if useResnet:
        model = torch.load('40incepmodel.pkl')
    else:
        model = torch.load('160longmodel.pkl')
        # model = torch.load('89533cnn4model.pkl')
    model.eval()
    




if useOfficialData:
    if ModelFusion:
        test_dataset = datasets.FashionMNIST(root = './data/',
                                                train = False,
                                                transform = transforms.Compose([
                                                transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5], [0.5]),
                                                ]))
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                outputs1 = model1(images)
                outputs2 = model2(images)
                outputs3 = model3(images)
                outputs4 = model4(images)
                outputs1 = F.softmax(outputs1,dim=1)
                outputs2 = F.softmax(outputs2,dim=1)
                outputs3 = F.softmax(outputs3,dim=1)
                outputs4 = F.softmax(outputs4,dim=1)
                outputs = (outputs1+outputs2+outputs3+outputs4)/4.0
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    else:       
        test_dataset = datasets.FashionMNIST(root = './data/',
                                                train = False,
                                                transform = transforms.Compose([
                                                transforms.Resize(img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5], [0.5]),
                                                ]))
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
else:
    if ModelFusion:
        test_data = npy.load('test.npy')
        test_label = pd.read_csv('train.csv')
        test_loader = Data.DataLoader(dataset=MyDataset(test_data,test_label,transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5]),
])),batch_size=batch_size,shuffle=False)

        pred_result=[]
        with torch.no_grad():
            for i,(images,labels) in enumerate(test_loader):
                images = Variable(images).cuda()
                outputs1 = model1(images)
                outputs2 = model2(images)
                outputs3 = model3(images)
                outputs4 = model4(images)
                outputs1 = F.softmax(outputs1,dim=1)
                outputs2 = F.softmax(outputs2,dim=1)
                outputs3 = F.softmax(outputs3,dim=1)
                outputs4 = F.softmax(outputs4,dim=1)
                outputs = (outputs1+outputs2+outputs3+outputs4)/4.0

                _,predicted = outputs.max(1)
                pred_list = list(predicted.cpu().numpy())
                pred_result.extend(pred_list)

            predict = npy.array(pred_result)
            predict = predict.reshape(-1,1)

            result = npy.concatenate((npy.arange(0,5000).reshape(-1,1),predict),axis=1)
            test_result = pd.DataFrame(result,columns=['image_id','label'])
            # test_result = pd.DataFrame({'image_id':npy.arange(0,4999).reshape(-1,1),'label':predict},)
            test_result.to_csv('test.csv',index=False)
    else:
        test_data = npy.load('test.npy')
        test_label = pd.read_csv('train.csv')
        test_loader = Data.DataLoader(dataset=MyDataset(test_data,test_label),batch_size=batch_size,shuffle=False)

        pred_result=[]
        with torch.no_grad():
            for i,(images,labels) in enumerate(test_loader):
                images = Variable(images).cuda()
                outputs = model(images)

                _,predicted = outputs.max(1)
                pred_list = list(predicted.cpu().numpy())
                pred_result.extend(pred_list)

            predict = npy.array(pred_result)
            predict = predict.reshape(-1,1)

            result = npy.concatenate((npy.arange(0,5000).reshape(-1,1),predict),axis=1)
            test_result = pd.DataFrame(result,columns=['image_id','label'])
            # test_result = pd.DataFrame({'image_id':npy.arange(0,4999).reshape(-1,1),'label':predict},)
            test_result.to_csv('test.csv',index=False)