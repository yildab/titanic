import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

batch_size = 50
learning_rate = 0.01
epochs = 10
interval = 10

raw_train_data = pd.read_csv('train.csv')
raw_test_data = pd.read_csv('test.csv')
data = raw_train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
data = data.replace(['male', 'female'], [0, 1])
data['Age'].fillna(value=data['Age'].median(), inplace = True)
data['Fare'].fillna(value=data['Fare'].median(), inplace = True)

class TrainingDataset(Dataset):
    def __init__(self):
        self.data = raw_train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
        self.data = self.data.replace(['male', 'female'], [0, 1])
        self.data['Age'].fillna(value=data['Age'].median(), inplace = True)
        self.data['Fare'].fillna(value=data['Fare'].median(), inplace = True)
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.from_numpy(self.data.iloc[index, 1:].values)
        y = torch.from_numpy(np.asarray([self.data.iloc[index, 0]]))
        if y[0] == 0:
            y = torch.tensor([1,0])
        else:
            y = torch.tensor([0,1])
        return x, y

class TestDataset(Dataset):
    def __init__(self):
        self.data = raw_train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
        self.data = self.data.replace(['male', 'female'], [0, 1])
        self.data['Age'].fillna(value=data['Age'].median(), inplace = True)
        self.data['Fare'].fillna(value=data['Fare'].median(), inplace = True)
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = torch.from_numpy(self.data.iloc[index, :].values)
        return x

train_data = TrainingDataset()
test_data = TestDataset()

train_loader = DataLoader(dataset= train_data, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size = batch_size, shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, ),
        )

    def forward(self, x):
        return self.layers(x.float())

titanicNet = Net()

optimizer = optim.SGD(titanicNet.parameters(), lr=learning_rate, momentum=0.9)

criterion = nn.NLLLoss()

for epoch in range(epochs):
    for batch_number, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        net_output = titanicNet(data)
        loss = criterion(net_output, target)
        loss.backward()
        optimizer.step()
        if batch_number % interval == 10:
            print('Epoch:', epoch, 'Loss = ', loss.item())
