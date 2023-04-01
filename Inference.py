import numpy as np
from sklearn.preprocessing import MinMaxScaler  
import os
import itertools
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import grad
from torch.utils.data import DataLoader,Dataset, TensorDataset
seed = 42
import torch
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout
import torchvision.models as models
import torch.optim as optim

from Model import MultiTask
from multi_output_data_prep import MultiOutputDataset
from losses import XSigmoidLoss, LogCoshLoss, AlgebraicLoss


X = np.load('X_inference.npy')
data = torch.tensor(X, dtype=torch.float)

def Inference():
    model = MultiTask()
    PATH = "Saved_Trained_Model.pth"
    model.load_state_dict(torch.load(PATH))

    model.eval()
    with torch.no_grad():
        output1, output2, output3, output4 = model(data)
    
    print('Task 1 Predictions:', output1)
    print('Task 2 Predictions:', output2)
    print('Task 3 Predictions:', output3)
    print('Task 4 Predictions:', output4)