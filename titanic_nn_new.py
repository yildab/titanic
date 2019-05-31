import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import csv

batch_size = 100
learning_rate = 0.01
epochs = 150
interval = 10

raw_test_data = pd.read_csv('test.csv')

def preprocess_features(dataframe):
    #drop useless features
    drop = ["PassengerId","Name", "Ticket", "Cabin", "Survived"]
    df = dataframe.drop(drop, axis=1)

    #deal with empty cells
    df['Age'].fillna(value=df['Age'].median(), inplace=True)
    df['Fare'].fillna(value=df['Fare'].median(), inplace=True)
    df['Pclass'].fillna(value=df['Pclass'].mode(), inplace=True)
    df['Sex'].fillna(value=df['Sex'].mode(), inplace=True)
    df['Embarked'].fillna(value=df['Embarked'].mode(), inplace=True)

    #replace categories with dummy variables
    categories = ["Pclass", "Sex", "Embarked"]
    df = pd.get_dummies(df, columns=categories)

    return df

def preprocess_test(dataframe):
    #drop useless features
    drop = ["PassengerId","Name", "Ticket", "Cabin"]
    df = dataframe.drop(drop, axis=1)

    #deal with empty cells
    df['Age'].fillna(value=df['Age'].median(), inplace=True)
    df['Fare'].fillna(value=df['Fare'].median(), inplace=True)
    df['Pclass'].fillna(value=df['Pclass'].mode(), inplace=True)
    df['Sex'].fillna(value=df['Sex'].mode(), inplace=True)
    df['Embarked'].fillna(value=df['Embarked'].mode(), inplace=True)

    #replace categories with dummy variables
    categories = ["Pclass", "Sex", "Embarked"]
    df = pd.get_dummies(df, columns=categories)

    return df

def preprocess_target(target):
    target = pd.get_dummies(target, columns=["Survived"])
    return target

#xtrain = preprocess_features(raw_xtrain)
#ytrain = preprocess_target(raw_ytrain)
#print(xtrain.head(), xtrain.shape)
#print(ytrain.head(), ytrain.shape)

class TrainDataset(Dataset):
    def __init__(self):
        raw_xtrain = pd.read_csv('train.csv', delimiter=',', skipinitialspace=True)
        self.x = preprocess_features(raw_xtrain)
        self.y = preprocess_target(raw_xtrain["Survived"])
        self.to_Tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        x = np.asarray(self.x.iloc[index, :])
        y = np.asarray(self.y.iloc[index, :])
        tensor_x = torch.from_numpy(x)
        tensor_y = torch.from_numpy(y)
        return tensor_x, tensor_y

train_data = TrainDataset()

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 400),
            nn.ReLU(),
            nn.Linear(400, 2),
            nn.Softmax()
        )
    def forward(self,x):
        return self.layers(x.float())

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

titanicNet = Net()
titanicNet.apply(init_weights)

optimizer = optim.SGD(titanicNet.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.MSELoss()

for epoch in range(epochs):
    correct_count = 0
    for batch_number, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        net_output = titanicNet(data)
        loss = criterion(net_output, target.float())
        loss.backward()
        optimizer.step()
        if batch_number % interval == 0:
            print('Epoch:', epoch, 'Loss = ', loss.item())

## TEST DATA!

class TestDataset(Dataset):
    def __init__(self):
        raw_xtrain = pd.read_csv('train.csv', delimiter=',', skipinitialspace=True)
        self.x = preprocess_test(raw_test_data)
        self.to_Tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.x)

    def __getitem__(self,index):
        x = np.asarray(self.x.iloc[index, :])
        tensor_x = torch.from_numpy(x)
        return tensor_x

test_data = TestDataset()

test_loader = DataLoader(dataset=test_data, batch_size=418, shuffle=False)

with open("submission.csv", 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['PassengerId', 'Survived'])

    for batch_number, data in enumerate(test_loader):
        net_output = titanicNet(data)
        for id, data in enumerate(net_output):
            PassID = id + 1
            value, index = data.max(0)
            survival = index.item()
            writer.writerow([PassID+891, survival])
