import torch
from torch import nn
from torch.nn import functional as F
import torch


class MultiTask(nn.Module):
    def __init__(self):
        super(MultiTask,self).__init__()
        self.conv1 = nn.Conv1d(12,256, 5)
        self.conv2 = nn.Conv1d(256, 196, 5)
        self.conv3 = nn.Conv1d(196, 128, 5)
        
        self.bn1 = nn.BatchNorm1d(256)
        #self.bn2 = nn.BatchNorm1d(196)
        
        self.maxpool1 = nn.MaxPool1d(1, stride=2)
        x = torch.randn(1, 12, 256)
        self._to_linear = None
        self.convs(x)
    
        self.fc = nn.Linear(self._to_linear, 32)
        self.fc_1 = nn.Linear(32, 16)
        self.fc1 = nn.Linear(16,1)
        self.fc2 = nn.Linear(16,1)
        self.fc3 = nn.Linear(16,1)
        self.fc4 = nn.Linear(16,1)
    
    
    def convs(self, x):

        x = self.maxpool1(self.bn1(F.sigmoid(self.conv1(x))))
        x = self.maxpool1((F.relu(self.conv2(x))))
        x = self.maxpool1(F.relu(self.conv3(x)))
       
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]
            #self._to_linear = x[0].shape[0]*x[0].shape[1]
        return x
        
    def forward(self, x):

        x = torch.reshape(x, (-1, 12, 256))
        x = self.maxpool1(self.bn1(F.relu(self.conv1(x))))
        x = self.maxpool1((F.relu(self.conv2(x))))
        x = self.maxpool1(F.relu(self.conv3(x)))
        
        x = x.view(-1, self._to_linear)
        x = F.tanh(self.fc(x))
        x = F.tanh(self.fc_1(x))
        
        F1 = F.sigmoid(self.fc1(x))
        F2 = F.sigmoid(self.fc2(x))
        F3 = F.sigmoid(self.fc3(x))
        F4 = F.sigmoid(self.fc4(x))

        return [F1, F2, F3, F4]   
