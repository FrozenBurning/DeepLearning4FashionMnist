import torch     
import torch.nn as nn
import torch.nn.functional as F
img_size = 28

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
            nn.BatchNorm2d(128),
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
