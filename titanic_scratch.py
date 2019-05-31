import numpy as np
import pandas as pd

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

def sigmoid(s):
    return 1/(1+np.exp(-s))

def d_sigmoid(s):
    return sigmoid(s)*(1-sigmoid(s))

def softmax(s):
    exps = np.exp(s - np.max(s,axis=1,keepdims=True))
    return exps/np.sum(exps,axis=1,keepdims=True)

def cross_entropy(yhat, y):
    number_samples = y.size
    loss = 0
    for i in range(number_samples):
        target = y[i]
        prediction = yhat[i]
        if prediction[target] != 0:
            logP = - np.log(prediction[target])
            loss += logP
    average_loss = loss/number_samples
    return average_loss

class Titanic_Net:
    def __init__(self, input_size, output_size, hidden_size, rate):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.W3 = np.random.randn(hidden_size, output_size)

        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(hidden_size)
        self.b3 = np.zeros(output_size)

        self.learningRate = rate

        self.prediction = None

    def feedforward(self, X, y):

        if len(X.shape) == 1:
            x_temp = [X]
            self.input = np.array(x_temp)
        else:
            self.input = X

        self.target = np.eye(self.output_size)[y]

        self.a1 = np.dot(self.input, self.W1) + self.b1
        self.h1 = sigmoid(self.a1)
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = sigmoid(self.a2)
        self.a3 = np.dot(self.h2, self.W3) + self.b3
        self.prediction = softmax(self.a3)

        return self.prediction

    def predict(self):
        if len(self.target) == 1:
            y_temp = [self.target]
            self.target = np.array(y_temp)
            for pred, targ in zip(self.prediction, self.target):
                prediction = np.argmax(pred)
                print("Prediction:", prediction, ". Target:", np.argmax(targ))

    def backprop(self):
        N_samples = self.prediction.shape[0]

        self.dJ_da3 = self.prediction - self.target
        self.dJ_dh2 = np.dot(self.dJ_da3, self.W3.T)
        self.dJ_da2 = self.dJ_dh2 * d_sigmoid(self.h2)
        self.dJ_dh1 = np.dot(self.dJ_da2, self.W2.T)
        self.dJ_da1 = self.dJ_dh1 * d_sigmoid(self.h1)

        self.avg_dJ_dW1 = np.dot(self.input.T, self.dJ_da1) /N_samples
        self.avg_dJ_dW2 = np.dot(self.h1.T, self.dJ_da2) /N_samples
        self.avg_dJ_dW3 = np.dot(self.h2.T, self.dJ_da3) /N_samples
        self.avg_dJ_db3 = np.sum(self.dJ_da3)    /N_samples
        self.avg_dJ_db2 = np.sum(self.dJ_da2)    /N_samples
        self.avg_dJ_db1 = np.sum(self.dJ_da1)    /N_samples

        return [self.avg_dJ_dW1, self.avg_dJ_dW2, self.avg_dJ_dW3,
            self.avg_dJ_db1, self.avg_dJ_db2, self.avg_dJ_db3]

    def learning_step(self):
        self.W1 -= self.learningRate * self.avg_dJ_dW1
        self.W2 -= self.learningRate * self.avg_dJ_dW2
        self.W3 -= self.learningRate * self.avg_dJ_dW3
        self.b1 -= self.learningRate * self.avg_dJ_db1
        self.b2 -= self.learningRate * self.avg_dJ_db2
        self.b3 -= self.learningRate * self.avg_dJ_db3

model = Titanic_Net(12, 2, 10, 0.01)

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

train_data = 
