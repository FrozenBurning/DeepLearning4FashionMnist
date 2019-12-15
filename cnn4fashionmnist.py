
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

class MyDataset(Data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        tmp_mat = (self.images[index]).reshape(28,28)
        img, target = tmp_mat[npy.newaxis,:], self.labels['label'][self.labels['image_id'][index]]
        img = img.astype(npy.float32)
        # print(img)
        # print("label: "+str(target))
        return img, target

    def __len__(self):
        return len(self.images)



train_data = npy.load('train.npy')
train_label = pd.read_csv('train.csv')
test_data = npy.load('test.npy')
test_label = pd.read_csv('train.csv')


batch_size=128
n_iters=18000
num_epochs=n_iters/(30000/batch_size)
num_epochs=int(num_epochs)



train_loader = Data.DataLoader(dataset=MyDataset(train_data,train_label),batch_size=batch_size,shuffle=True)
test_loader = Data.DataLoader(dataset=MyDataset(test_data,test_label),batch_size=batch_size,shuffle=False)

class CNNModule(nn.Module):
    def __init__(self):
        super (CNNModule,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.dropout1=nn.Dropout(p=0.5)
        self.relu1=nn.ReLU()
        self.cnn1_bn = nn.BatchNorm2d(16)
        nn.init.xavier_uniform_(self.cnn1.weight,gain=math.sqrt(2))


        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.dropout2=nn.Dropout(p=0.2)
        self.relu2=nn.ReLU()
        self.cnn2_bn=nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn2.weight,gain=math.sqrt(2))

        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.dense1 = nn.Linear(32*7*7,50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.fcl=nn.Linear(50,10)
        # self.fcl=nn.Linear(32*7*7,10)
    def forward(self,x):
        out=self.cnn1_bn(F.relu(self.cnn1(x)))
        # out=self.relu1(out)
        #print ("CNN1")
        #print (out.size())

        # out=self.cnn1_bn(out)
        # out=self.dropout1(out)
        
        out=self.maxpool1(out)
        #print ("Maxpool1")
        #print (out.size())
        
        out=self.cnn2_bn(F.relu(self.cnn2(out)))
        # out=self.relu2(out)
        #print ("CNN2")
        #print (out.size())

        # out = self.cnn2_bn(out)

        out=self.dropout2(out)

        out=self.maxpool2(out)
        #print ("Maxpool2")
        #print (out.size(0))
        out = F.relu(out)
        out = out.view(out.size(0),-1)
        out=F.relu(self.dense1_bn(self.dense1(out)))

        # out=out.view(out.size(0),-1)

        out=F.relu(self.fcl(out))

        return out
model=CNNModule().cuda()
criterion=nn.CrossEntropyLoss()
learning_rate=0.003
# optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.001)


losses = [] 
acces = []
eval_losses = []
eval_acces = []


iter=0
# num_epochs = 1
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    model.train()
    for i,(images,labels) in enumerate (train_loader):
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
        # if iter%500==0:
        #     correct=0
        #     total=0
        #     for images,labels in test_loader:
        #         images=Variable(images)

        #         outputs=model(images)
                
        #         _,predicted=torch.max(outputs.data,1)
        #         total+=labels.size(0)
        #         correct+=(predicted==labels).sum()
        #     accuracy= (100.0* correct)/(total)
        #     print("Iteration:"+str(iter)+"  Loss:"+str(loss)+"  Accuracy:"+str(accuracy))

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
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' '+str(train_acc / len(train_loader)))

pred_result=[]
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

# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
 
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         #x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x

# model = Model().cuda() #实例化卷积层
# loss = nn.CrossEntropyLoss() #损失函数选择，交叉熵函数
# optimizer = optim.SGD(model.parameters(),lr = 0.05)
# num_epochs = 30


# losses = [] 
# acces = []
# eval_losses = []
# eval_acces = []

# for echo in range(num_epochs):
#     train_loss = 0   #定义训练损失
#     train_acc = 0    #定义训练准确度
#     model.train()    #将网络转化为训练模式
#     for i,(X,label) in enumerate(train_loader):     #使用枚举函数遍历train_loader
#         # X = X.view(-1,28,28)       #X:[64,1,28,28] -> [64,784]将X向量展平
#         X = Variable(X).cuda()          #包装tensor用于自动求梯度
#         label = Variable(label).cuda()
#         out = model(X)           #正向传播
#         lossvalue = loss(out,label)         #求损失值
#         optimizer.zero_grad()       #优化器梯度归零
#         lossvalue.backward()    #反向转播，刷新梯度值
#         optimizer.step()        #优化器运行一步，注意optimizer搜集的是model的参数
        
#         #计算损失
#         train_loss += float(lossvalue)      
#         #计算精确度
#         _,pred = out.max(1)
#         num_correct = (pred == label).sum()
#         acc = int(num_correct) / X.shape[0]
#         train_acc += acc

#     losses.append(train_loss / len(train_loader))
#     acces.append(train_acc / len(train_loader))
#     print("echo:"+' ' +str(echo))
#     print("lose:" + ' ' + str(train_loss / len(train_loader)))
#     print("accuracy:" + ' '+str(train_acc / len(train_loader)))