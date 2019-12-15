
import pandas as pd
import torch     
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,models
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
import offi_resnet
import sys
from torchsummary import summary
from myresnet18 import ResNet18

img_size = 28 #image-net input type

class MyDataset(Data.Dataset):
    def __init__(self, images, labels,transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform


    def __getitem__(self, index):#返回的是tensor
        tmp_mat = (self.images[index]).reshape(img_size,img_size)
        # tmp_mat = self.images[index]
        # img, target = tmp_mat, self.labels['label'][self.labels['image_id'][index]]
        img,target =tmp_mat,self.labels[index]
        # img = tmp_mat
        # target = self.labels.iloc[int(index)]['label']
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


def Preproc(raw_data,raw_label,isTrain=True):
    if isTrain:
        result = npy.zeros((raw_data.shape[0]*6,raw_data.shape[1]*int(img_size/28)*int(img_size/28)))
        # tmp_label = raw_label.values
        # tmp_label = tmp_label[:,1]
        tmp_label = raw_label
        result_label = npy.concatenate((tmp_label,tmp_label))#*2
        result_label = npy.concatenate((result_label,tmp_label))#*3
        result_label = npy.concatenate((result_label,result_label))#*6
        for i in range(raw_data.shape[0]):
            source_data = (raw_data[i]).reshape(28,28)

            tmp_mat = cv2.resize(source_data,(img_size,img_size))
            tmp_mat = npy.fliplr(tmp_mat)
            target_mat = tmp_mat.reshape(1,img_size*img_size)
            result[i,:] = (cv2.resize(source_data,(img_size,img_size))).reshape(1,img_size*img_size)
            result[i+raw_data.shape[0]*1,:] = target_mat 

            tmp_mat = cv2.resize(source_data,(img_size,img_size))
            tmp_mat = Image.fromarray(npy.uint8(tmp_mat))
            rot_mat = transforms.RandomRotation(20)(tmp_mat)
            rot_mat = npy.asarray(rot_mat)
            tmp_mat = rot_mat
            # kernel = npy.array([[0,1,0],[1,-4,1],[0,1,0]])
            tmp_mat = tmp_mat.astype("float32")
            # tmp_mat = cv2.filter2D(tmp_mat,-1,kernel)
            # tmp_mat = cv2.blur(tmp_mat,(3,3))
            target_mat = tmp_mat.reshape(1,img_size*img_size)
            result[i+raw_data.shape[0]*2,:] = target_mat 

            tmp_mat = cv2.resize(source_data,(img_size,img_size))
            tmp_mat = tmp_mat.astype("float32")
            tmp_mat = cv2.blur(tmp_mat,(3,3))
            target_mat = tmp_mat.reshape(1,img_size*img_size)
            result[i+raw_data.shape[0]*3,:] = target_mat 

            tmp_mat = cv2.resize(source_data,(img_size,img_size))
            tmp_mat = Image.fromarray(npy.uint8(tmp_mat))
            tmp_tensor = transforms.ToTensor()(tmp_mat)
            tmp_tensor = transforms.RandomErasing()(tmp_tensor)
            tmp_mat = tmp_tensor.numpy()
            tmp_mat = tmp_mat[-1,:,:]
            tmp_mat = tmp_mat.astype("float32")
            target_mat = tmp_mat.reshape(1,img_size*img_size)
            result[i+raw_data.shape[0]*4,:] = target_mat

            tmp_mat = cv2.resize(source_data,(img_size,img_size))
            tmp_mat = Image.fromarray(npy.uint8(tmp_mat))
            tmp_mat = transforms.RandomCrop(img_size)(tmp_mat)
            tmp_mat = npy.asarray(tmp_mat)
            tmp_mat = tmp_mat.astype("float32")
            target_mat = tmp_mat.reshape(1,img_size*img_size)
            result[i+raw_data.shape[0]*5,:] = target_mat
    else:
        result = npy.zeros((raw_data.shape[0],raw_data.shape[1]*int(img_size/28)*int(img_size/28)))
        tmp_label = raw_label
        result_label = tmp_label
        for i in range(raw_data.shape[0]):
            source_data = (raw_data[i]).reshape(28,28)
            result[i,:] = (cv2.resize(source_data,(img_size,img_size))).reshape(1,img_size*img_size)
    return result,result_label


raw_train_data = npy.load('train_set.npy')
raw_train_label = npy.load('train_set_label.npy')

# train_data,train_label = raw_train_data,raw_train_label
train_data,train_label = Preproc(raw_train_data,raw_train_label)
raw_valid_data = npy.load('valid_set.npy')
raw_valid_label = npy.load('valid_set_label.npy')
valid_data,valid_label = Preproc(raw_valid_data,raw_valid_label,False)



batch_size=64

train_loader = Data.DataLoader(dataset=MyDataset(train_data,train_label,transform = transforms.Compose([
  transforms.RandomRotation(20),
  transforms.RandomCrop(img_size),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5]),
])),batch_size=batch_size,shuffle=True)
valid_loader = Data.DataLoader(dataset=MyDataset(valid_data,valid_label,transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5]),
])),batch_size=batch_size,shuffle=True)

class pretrained_res18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,3,kernel_size=1,padding=2,stride=1)
        self.res = models.resnet50(pretrained=False)
        self.bn = nn.BatchNorm1d(1000)
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(1000,4096)
        self.d_bn = nn.BatchNorm1d(4096)
        self.d_drop = nn.Dropout()
        self.fc = nn.Linear(4096,10)
        # for param in self.res.layer1.parameters():
        #     param.requires_grad = False
        # for param in self.res.layer2.parameters():
        #     param.requires_grad = False
        # for param in self.res.layer3.parameters():
        #     param.requires_grad = False
        # for param in self.res.layer4.parameters():
        #     param.requires_grad = False
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.res(out)
        out = F.relu(out)
        out = self.bn(out)
        out = self.dropout(out)
        out = self.dense(out)
        out = F.relu(out)
        out = self.d_bn(out)
        out = self.d_drop(out)
        out = self.fc(out)
        return out


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.filen = filename

    def write(self, message):
	    self.terminal.write(message)
	    with open(self.filen,'a') as f:
             f.write(message)

    def flush(self):
	    pass

sys.stdout = Logger('pretrain_resnet50.log', sys.stdout)


# model = pretrained_res18().cuda()
model = torch.load('20advancedres50model.pkl')
# model = offi_resnet.resnet18(pretrained=False,num_classes=10).cuda()
# model = torch.load('maybegoodresnet_model.pkl')
weights = [2.0,0.8,1.5,0.8,1.5,1.2,2.0,1.2,1,1.2]
class_weights = torch.FloatTensor(weights).cuda()
criterion=nn.CrossEntropyLoss(weight=class_weights)
learning_rate=0.001
# learning_rate = 3e-4
# optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=5e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch > 200:
        lr = 1e-5
    else:
            lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

losses = [] 
acces = []
eval_losses = []
eval_acces = []

summary(model,(1,28,28))
iter=0
num_epochs =300
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    adjust_learning_rate(optimizer,epoch)
    # model.train()
    for i,(images,labels) in enumerate (train_loader):
        model.train(True)
        images=Variable(images).cuda()
        labels=Variable(labels).cuda()

        optimizer.zero_grad()
        outputs=model(images)
        # print(outputs.size())
        # print(labels.size())
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        iter+=1
        if iter%500==0:
            model.eval()
            correct=0
            total=0
            for i,(valid_images,valid_labels) in enumerate(valid_loader):
                valid_images=Variable(valid_images).cuda()
                valid_labels = Variable(valid_labels).cuda()
                test_outputs=model(valid_images)
                
                _,predicted=test_outputs.max(1)
                total+=valid_labels.size(0)
                correct+=(predicted==valid_labels).sum()
            accuracy= (100.0* correct)/(total)
            print("Iteration:"+str(iter)+"  Loss:"+str(loss)+"  Accuracy:"+str(accuracy))

        #计算损失
        train_loss += float(loss)      
        #计算精确度
        _,pred = outputs.max(1)
        # _,pred = torch.max(outputs.data,1)
        num_correct = (pred == labels).sum()
        acc = int(num_correct) / images.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print("echo:"+' ' +str(epoch))
    print("loss:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' '+str(train_acc / len(train_loader)))
    if epoch%20 == 0:
        model_checkpoint = str(epoch)+"advancedres50model.pkl"
        torch.save(model,model_checkpoint)