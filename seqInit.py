import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data.csv', usecols=[1])

# numpy -> tensor

def toTs(x): return torch.from_numpy(x)

# Data processing
data = data.dropna()
dataSet = data.values
dataSet = dataSet.astype('float32')
# print("data Shape:", dataSet.shape)

# Data normalization
def MinMaxScaler(X):
    mx, mi = np.max(X), np.min(X)
    X_std = (X-mx)/(mx-mi)
    return X_std

# devide dataset into train and val
dataSet = MinMaxScaler(dataSet)
train = dataSet[:12*10]
val = dataSet

input_size = 3
