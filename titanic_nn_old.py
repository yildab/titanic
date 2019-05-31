# Dataloader Practice, sources:
#       https://www.youtube.com/watch?v=zN49HdDxHi8

# Dataset: Titanic: Machine Learning from Disaster
#       https://www.kaggle.com/c/titanic
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

batch_size = 50
learning_rate = 0.01
epochs = 10
log_interval = 10

class TitanicTrainDataset(Dataset):
    '''Dataset of training data for Titanic survival. '''
    '''Columns PassengerID (0), Name (4), Ticket (8) are non-relevant features.'''
    '''Target (survived or not) is column 1.'''
    def __init__(self):
        data = np.loadtxt('train.csv', delimiter=',')
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 1:-1])
        self.y_data = torch.from_numpy(data[:, 1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TitanicTestDataset(Dataset):
    '''Dataset of training data for Titanic survival. '''
    '''Columns PassengerID (0), Name (4), Ticket (8) are non-relevant features.'''
    '''Target (survived or not) is column 1.'''
    def __init__(self):
        data = np.loadtxt('test.csv', delimiter=',', skiprows=1, usecols=(1, 2, 4, 5, 6, 7, 9, 10, 11))
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 1:-1])
        self.y_data = torch.from_numpy(data[:, 1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

titanic_train_data = TitanicTrainDataset()
titanic_test_data = TitanicTestDataset()

# FEATURE SELECTION

# # Manual removal: PassengerID, Name, Ticket
# def ManualFeatureSelect(ds, features):
#     for i in features:
#         ds.x_data = torch.cat(ds.x_data[0:i], ds.x_data[i+1:])
# ManualFeatureSelect(titanic_data)

train_loader = DataLoader(dataset=titanic_train_data, batch_size = batch_size, shuffle = True)

print(TitanicTrainDataset.__len__())
