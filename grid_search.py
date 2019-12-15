
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
import sys

img_size = 28

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
            img,target = self.transform(img,target)
        else:
            img = img[npy.newaxis,:]
        return img, target

    def __len__(self):
        return len(self.images)


def Preproc(raw_data,raw_label):
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
        rot_mat = transforms.RandomRotation(10)(tmp_mat)
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

    return result,result_label

class CNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32,32,kernel_size=3,padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.25),

            nn.Conv2d(64,128,kernel_size=3,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*int((img_size-12)/4)*int((img_size-12)/4), 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, 10),       
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

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

sys.stdout = Logger('grid_search4cnn.log', sys.stdout)


raw_train_data = npy.load('train_set.npy')
raw_train_label = npy.load('train_set_label.npy')
train_data,train_label = Preproc(raw_train_data,raw_train_label)

valid_data = npy.load('valid_set.npy')
valid_label = npy.load('valid_set_label.npy')


batch_size_list=[32,64,128,256,512]
num_epochs_list = [20,30,40,50,60,70,80,90,100,200]

for batch_size in batch_size_list:
    for num_epochs in num_epochs_list:
        print("*********batch_size=",batch_size,",num_epoch=",num_epochs,"*********")
        train_loader = Data.DataLoader(dataset=MyDataset(train_data,train_label),batch_size=batch_size,shuffle=True)
        valid_loader = Data.DataLoader(dataset=MyDataset(valid_data,valid_label),batch_size=batch_size,shuffle=True)

        model=CNNModule().cuda()
        model.apply(weight_init)
        # model = torch.load('good4CNN_model.pkl')
        criterion=nn.CrossEntropyLoss()
        learning_rate=0.001
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


        losses = [] 
        acces = []
        eval_losses = []
        eval_acces = []

        accuracy = 0
        iter=0
        for epoch in range(num_epochs):
            train_loss = 0
            train_acc = 0
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

                model.train(True)
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
        
        valid_acc = accuracy.cpu().item()
        if valid_acc> 90:
            model_name = "model_bs" + str(batch_size) + "_epc" + str(num_epochs)+"_acc"+str(valid_acc)+".pkl"
            torch.save(model,model_name)