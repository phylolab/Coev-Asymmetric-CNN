import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt


"""NUMBER_FILTERS = 400
BATCH_SIZE = 64
ROW_SIZE = 199 #100
COL_SIZE = 396 #672"""

### Custom dataloaders
class trainData(Dataset):
    
    def __init__(self, X_data, y_data, label_data):
        self.X_data = X_data
        self.y_data = y_data
        self.label_data = label_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.label_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
class coevClassifier_Separate(nn.Module):
    def __init__(self, NUMBER_FILTERS, ROW_SIZE, COL_SIZE):
        super(coevClassifier_Separate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = NUMBER_FILTERS, 
                               kernel_size = (ROW_SIZE,1))
        
        self.conv2 = nn.Conv2d(in_channels = NUMBER_FILTERS, out_channels = NUMBER_FILTERS, 
                               kernel_size = (1,COL_SIZE))

        self.fc1 = nn.Linear(NUMBER_FILTERS, 100)
        self.fc1_drop = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(100, 2)


    def forward(self, x):
        #print(x.shape)
        #x2 = torch.sum(x, dim=1, keepdim=True)
        
        #x = x.type(torch.LongTensor)
        x2 = torch.from_numpy(np.expand_dims(x, axis=1))
        #print(x2.shape)
        
        x_conv1 = F.relu(self.conv1(x2))
        #print(x_conv1.shape)
        
        x2_conv1 = x_conv1
        
        #print(x2_conv1.shape)
        #print(x2_conv1)
        
        x_conv2 = F.relu(self.conv2(x2_conv1))
        
        x_conv2 = x_conv2.view(x_conv2.size(0), -1)
        
        x_final = F.relu(self.fc1(x_conv2))
        x_final = self.fc1_drop(x_final)
        x_final = self.fc3(x_final)
        
        return x_final

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc
    
def train_network(NUMBER_EPOCHS, train_loader, val_loader, model, criterion, optimizer):
    print("Begin training.")
    
    accuracy_stats = {
    'train': [],
    "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    
    for e in range(1, NUMBER_EPOCHS):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for batch_idx, (X_train_batch, y_train_batch, filename_train_batch) in enumerate(train_loader):
            #for X_train_batch, y_train_batch, filename_train_batch in train_loader:

            optimizer.zero_grad()
            y_train_pred = model(X_train_batch.type(torch.float))
            #print(y_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch.type(torch.LongTensor))
            #train_loss = train_loss + l2_lambda * l2_norm
            train_acc = binary_acc(y_train_pred, y_train_batch.type(torch.LongTensor))

            # Backpropagation
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for batch_idx, (X_val_batch, y_val_batch, filename_val_batch) in enumerate(val_loader):

                #for X_val_batch, y_val_batch, filename_val_batch in val_loader:
                #X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch.type(torch.float)) #.squeeze()
                #y_val_pred = torch.unsqueeze(y_val_pred, 0)
                val_loss = criterion(y_val_pred, y_val_batch.type(torch.LongTensor))
                val_acc = binary_acc(y_val_pred, y_val_batch.type(torch.LongTensor))
                val_epoch_loss += train_loss.item()
                val_epoch_acc += train_acc.item()
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
        print(f'Epoch {e+0:02}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
        
    return model, loss_stats, accuracy_stats


def test_model(model, test_loader):
    y_pred_list = []
    y_true_list = []
    name_list = []
    with torch.no_grad():
        for x_batch, y_batch, filename_batch in test_loader:
            #x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch.type(torch.float))
            y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
            name_list.append(filename_batch)
            #print(y_batch)

    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]
    name_list = [i[0] for i in name_list]
    
    return y_pred_list, y_true_list, name_list 


