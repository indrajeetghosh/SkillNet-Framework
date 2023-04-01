#import torch
from torch.utils.data import DataLoader,Dataset, TensorDataset


class MultiOutputDataset(Dataset):
    def __init__(self, X, y1, y2, y3, y4):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y1[idx], self.y2[idx], self.y3[idx], self.y4[idx]