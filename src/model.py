import torch
import config

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data.dataset as dataset


class ShapeDataset(dataset.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, item):
        target = torch.tensor(self.targets[item], dtype = torch.int)
        target= target.type(torch.LongTensor)
        image = torch.tensor(
            np.array(self.features.iloc[item]), dtype = torch.double
            )
        image = image.reshape(
            config.CHANNELS,
            config.IMAGE_H,
            config.IMAGE_W,
            )
        return (image, target)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (15,15))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 16, (4,4))
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 2 * 2, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 180)
        self.bn4 = nn.BatchNorm1d(180)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(180, 3)
        

    def forward(self, x):
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 2 * 2)
        x = self.dropout2(F.relu(self.bn3(self.fc1(x))))
        x = self.dropout3(F.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    print(Model())
