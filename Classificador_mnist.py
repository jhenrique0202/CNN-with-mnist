#import tensorflow as tf
#import datetime, os
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import requests
import gzip
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from torch import nn
from torch import optim
from torch.nn import functional as F
import torch

def get_data_set(mini_batch = 32, validation = False):
    image_size = 28
    num_images_train = 60000
    num_images_test = 10000

    PATH = Path(".", "data", "mnist")
    if not PATH.is_dir():
        PATH.mkdir(parents = True, exist_ok = True)
    
    URL = "http://yann.lecun.com/exdb/mnist/"
    train_x_file = "train-images-idx3-ubyte.gz"
    train_y_file = "train-labels-idx1-ubyte.gz"
    test_x_file = "t10k-images-idx3-ubyte.gz"
    test_y_file = "t10k-labels-idx1-ubyte.gz"
    
    for f in [train_x_file,train_y_file,test_x_file,test_y_file]:
        """if not (PATH / f).exists():
            content = requests.get(URL + f).content
            (PATH / f).open("wb").write(content)"""
        
        with gzip.open((PATH / f), "rb") as d:
            if f == train_x_file:
                d.read(16)
                buf = d.read(image_size * image_size * num_images_train)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                train_x = data.reshape(num_images_train, image_size*image_size)
            elif f == train_y_file:
                d.read(8)
                buf = d.read(num_images_train)
                train_y = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
                #train_y = data.reshape(data.shape[0],1)
                
                """first = True
                for i in data:
                    if first == True:
                        train_y = torch.zeros(1,10, dtype=torch.int64)
                        train_y[0,i] = 1
                        first = False
                    else:
                        y = torch.zeros(1,10, dtype=torch.int64)
                        y[0,i] = 1
                        train_y = torch.cat((train_y,y), 0)
                first = True
                for i in data:
                    if first == True:
                        train_y = torch.from_numpy(data)
                        first = False
                    else:
                        train_y = torch.cat((train_y,torch.from_numpy(data)), 0)"""
                
            elif f == test_x_file:
                d.read(16)
                buf = d.read(image_size * image_size * num_images_test)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
                test_x = data.reshape(num_images_test, image_size*image_size)
            elif f == test_y_file:
                d.read(8)
                buf = d.read(num_images_test)
                test_y = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
                #test_y = data.reshape(data.shape[0],1)

                """first = True
                for i in data:
                    if first == True:
                        test_y = torch.zeros(1,10, dtype=torch.int64)
                        test_y[0,i] = 1
                        first = False
                    else:
                        y = torch.zeros(1,10, dtype=torch.int64)
                        y[0,i] = 1
                        test_y = torch.cat((test_y,y), 0)
                first = True
                for i in data:
                    if first == True:
                        test_y = torch.from_numpy(data)
                        first = False
                    else:
                        test_y = torch.cat((test_y,torch.from_numpy(data)), 0)"""
            
    (train_x,train_y,test_x,test_y) = map(torch.tensor, (train_x,train_y,test_x,test_y))
    train_y.type(torch.int64)
    test_y.type(torch.int64)

    if validation:
        data_set_train = TensorDataset(train_x[:50000, : ],train_y[:50000])
        data_set_validation = TensorDataset(train_x[50000:, : ],train_y[50000:])
        data_set_test = TensorDataset(test_x, test_y)
        
        return (DataLoader(data_set_train, mini_batch, shuffle = True),
                DataLoader(data_set_train, len(data_set_train)),
                DataLoader(data_set_validation, len(data_set_validation)),
                DataLoader(data_set_test, len(data_set_test)))
    else:
        data_set_train = TensorDataset(train_x,train_y)
        data_set_test = TensorDataset(test_x, test_y)
    
        return (DataLoader(data_set_train, mini_batch, shuffle = True),
                DataLoader(data_set_train, len(data_set_train)),
                None,
                DataLoader(data_set_test, len(data_set_test)))
 

class Function_class(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function
    
    def forward(self, x):
        return self.function(x)

def Convert_array_to_image(x):
    return x.view(-1,1,28,28)

def Convert_image_to_array(x):
    size = x.size()[1:] 
    num_features = 1
    for s in size:
        num_features *= s
    return x.view(-1, num_features)

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Function_class(Convert_array_to_image),
            nn.Conv2d(1, 20, 5),
            nn.MaxPool2d(2,1),
            Function_class(Convert_image_to_array),
            nn.Linear(10580, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, 100),
            nn.Sigmoid(),
            nn.Linear(100,10))
    
    def forward(self,x):
        return self.layers(x)

def get_model(lr = 1e-3):
    model = Mnist_CNN()
    model = model.to(dev)
    optimizer = optim.SGD(model.parameters(), lr = lr)
    return model, optimizer

def fit(data_set_train, data_set_train_complete, data_set_validation, data_set_test, epoch = 60):
    model, optimizer = get_model()
    for e in range(epoch):
        for x,y in data_set_train:
            loss_function = nn.CrossEntropyLoss()

            loss = loss_function(model(x.to(dev)), y.to(dev))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        Evaluate_loss(model, loss_function, data_set_train_complete, "Loss/train", e)
        Evaluate_loss(model, loss_function, data_set_validation,"Loss/validation", e)
        
        Evaluate_accuracy(model, loss_function, data_set_train_complete, "Accuracy/train", e)
        Evaluate_accuracy(model, loss_function, data_set_validation, "Accuracy/validation", e)
        
    Evaluate_loss(model, loss_function, data_set_test, "Loss/test", e)
    Evaluate_accuracy(model, loss_function, data_set_test, "Accuracy/test", e)

def Evaluate_loss(model, loss_function, data_set, kind_data_set, epoch):
    for x,y in data_set:
        loss = loss_function(model(x.to(dev)), y.to(dev))
        graph.add_scalar(kind_data_set, loss, epoch)
        print(kind_data_set + ": " + str(loss), end = "\n")

def Evaluate_accuracy(model, loss_function, data_set, kind_data_set, epoch):
    for x,y in data_set:
        pred = model(x.to(dev))
        pred = torch.argmax(pred, dim = 1)
        
        accuracy = (pred == y.to(dev)).float().mean()
        graph.add_scalar(kind_data_set, accuracy, epoch)

        print(kind_data_set + ": " + str(accuracy), end = "\n")
                
graph = SummaryWriter()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_set_train, data_set_train_complete, data_set_validation, data_set_test = get_data_set(validation = True)
fit(data_set_train, data_set_train_complete, data_set_validation, data_set_test)
graph.close()
    
    
    
    