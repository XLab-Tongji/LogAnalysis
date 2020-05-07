import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os
from . import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_label(logkey_path, window_length,num_of_classes):
    f = open(logkey_path,'r')
    keys = f.readline().split()
    keys = list(map(int, keys))
    print(keys)
    length = len(keys)
    input_1 = np.zeros((length -window_length,1))
    output_1 = np.zeros(length -window_length,dtype=np.int)
    input_2 = np.zeros((length -window_length,num_of_classes))
    output = np.zeros(length -window_length,dtype=np.int)
    for i in range(0,length -window_length):
        for j in range(i,i+window_length):
            input_1[i][0] = keys[j]
            input_2[i][keys[j]-1] += 1
        output[i] = keys[i+window_length]-1
    new_input_1 = np.zeros((length -2*window_length+1,window_length,1))
    new_input_2 = np.zeros((length - 2 * window_length + 1, window_length, num_of_classes))
    for i in range(0,length -2*window_length+1):
        for j in range(i,i+window_length):
            new_input_1[i][j - i] = input_1[j]
            new_input_2[i][j-i] = input_2[j]
    new_output = output[window_length-1:]
    print(new_input_1.shape)
    print(new_input_2.shape)
    print(new_output.shape)
    dataset = TensorDataset(torch.tensor(new_input_1,dtype=torch.float),
                            torch.tensor(new_input_2,dtype=torch.float),torch.tensor(new_output,dtype=torch.long))
    return dataset

class Model(nn.Module):
    def __init__(self, input_size_0,input_size_1, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm0 = nn.LSTM(input_size_0, hidden_size, num_of_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size_1, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, out_size)


    def forward(self, input_0,input_1):
        h0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        c0_0 = torch.zeros(self.num_of_layers, input_0.size(0), self.hidden_size).to(device)
        out_0, _ = self.lstm0(input_0, (h0_0, c0_0))
        h0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        c0_1 = torch.zeros(self.num_of_layers, input_1.size(0), self.hidden_size).to(device)
        out_1, _ = self.lstm1(input_1, (h0_1, c0_1))
        multi_out = torch.cat((out_0[:, -1, :], out_1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out

def train_model(window_length, input_size_0,input_size_1, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory,logkey_path):
    # log setting
    log_directory = root_path + 'log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    print("Train num_classes: ", num_of_classes)
    model = Model(input_size_0,input_size_1, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    data_set = generate_label(logkey_path, window_length,num_of_classes)
    # create data_loader
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, quan, label) in enumerate(data_loader):
            seq = seq.clone().detach().view(-1, window_length, input_size_0).to(device)
            quan = quan.clone().detach().view(-1, window_length, input_size_1).to(device)
            output = model(seq,quan)

            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.6f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % 100 == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')

if __name__=='__main__':
    input_size_0 = 1
    input_size_1 = 61
    hidden_size = 30
    num_of_layers = 2
    num_of_classes = 61
    num_epochs = 100
    batch_size = 200
    window_length = 5
    train_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_train'
    test_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_test'
    train_root_path = '../../Data/FTTreeResult-HDFS/model_train/'
    label_file_name = '../../Data/FTTreeResult-HDFS/deeplog_files/HDFS_abnormal_label.txt'
    model_out_path = train_root_path + 'model_out/'
    train_model(window_length, input_size_0,input_size_1, hidden_size,
                num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path,
                model_out_path, train_logkey_path)