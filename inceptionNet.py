import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channel, n1_1, n3x3red, n3x3, n5x5red, n5x5, pool_plane):
        super(Inception, self).__init__()
        # first line
        self.branch1x1 = BasicConv2d(in_channel, n1_1, kernel_size=1)

        # second line
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channel, n3x3red, kernel_size=1),
            BasicConv2d(n3x3red, n3x3, kernel_size=3, padding=1)
        )

        # third line
        self.branch5x5 = nn.Sequential(
            BasicConv2d(in_channel, n5x5red, kernel_size=1),
            BasicConv2d(n5x5red, n5x5, kernel_size=5, padding=2)
        )

        # fourth line
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channel, pool_plane, kernel_size=1)
        )

    def forward(self, x):
        y1 = self.branch1x1(x)
        y2 = self.branch3x3(x)
        y3 = self.branch5x5(x)
        y4 = self.branch_pool(x)
        output = torch.cat([y1, y2, y3, y4], 1)
        return output

class SimpleNet(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleNet,self).__init__()

        self.conv1 = nn.Conv2d(1,64,kernel_size=1,padding=2,stride=1)

        self.incep1 = Inception(64,64,96,128,16,32,32)
        self.incep2 = Inception(256,128,128,192,32,96,64)

        self.maxpool = nn.MaxPool2d(2)

        self.incep3 = Inception(480,192,96,208,16,48,64)
        self.incep4 = Inception(512,160,112,224,24,64,64)

        self.avgpool = nn.AvgPool2d(7)

        self.dropout = nn.Dropout()

        self.fc = nn.Linear(2048,num_classes)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.incep1(x)
        x = self.incep2(x)
        x = self.maxpool(x)
        x = self.incep3(x)
        x = self.incep4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x