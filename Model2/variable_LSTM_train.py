#!/usr/bin/python
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
window_length = 4
hidden_size = 20
num_of_layers = 3
# For variable model，input size is the length of each log key's variable vector
# input_size = 2
num_epochs = 300
batch_size = 20  # 2048
# in_features = 10
# out_features = in_features

learning_rate = 0.01
root_path = '../Data/LogClusterResult-k8s/'
log_value_folder = root_path + 'logvalue/logvalue_train/'
model_output_directory = root_path + 'output/model2/'
log_template = 'model2_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, input):
        h0 = torch.zeros(self.num_of_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers, input.size(0), self.hidden_size).to(device)
        # h_n: hidden state h of last time step
        # c_n: hidden state c of last time step
        out, (h_n, c_n) = self.lstm(input, (h0, c0))
        # print('out size:')
        # print(out.size())
        # the output of final time step
        out = self.fc(out[:, -1, :])
        # print('out[:, -1, :]:')
        # print(out)
        return out


def generate(file_path):
    num_sessions = 0
    inputs = []
    outputs = []
    vectors = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(float, line.strip().split())))
            vectors.append(line)
            # For each log key's log parameter value vector，the length of these vector is same，the meaning of value at each position is same
            # eg: [log_vector1, log_vector2, log_vector3] --> log_vector4
            # so each element of inputs is a sequence，and each element of that sequence is a sequence too
            # nn's output is the prediction of parameter value vector
    for i in range(len(vectors) - window_length):
        inputs.append(vectors[i: i + window_length])
        outputs.append(vectors[i + window_length])
    # print(inputs)
    # print(inputs[0])
    data_set = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    if len(vectors) > 0:
        return data_set, len(vectors[0])
    else:
        return None, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=3, type=int)
    parser.add_argument('-hidden_size', default=20, type=int)
    parser.add_argument('-window_size', default=2, type=int)
    args = parser.parse_args()
    num_of_layers = args.num_layers
    hidden_size = args.hidden_size
    window_length = args.window_size
    file_names = os.listdir(log_value_folder)
    for i in range(len(file_names)):
        file_name = str(i+1) + ".log"
        train_data_set_name = log_value_folder + file_name
        validation_data_set_name = train_data_set_name

        train_data_set, input_size = generate(train_data_set_name)
        if input_size == 0:
            continue
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        validation_data_set, _ = generate(validation_data_set_name)
        validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        out_size = input_size
        model = Model(input_size, hidden_size, num_of_layers, out_size).to(device)
        
        writer = SummaryWriter(logdir =root_path + '/output/log2/' + str(i + 1) + '_' + log_template)

        # Loss and optimizer
        criterion = nn.MSELoss()  # use to predict
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_list = []  # Used to record the loss values during training. These values follow the Gaussian distribution.
        train_loss_list = []

        # Training and validating
        total_step = len(train_data_loader)
        for epoch in range(num_epochs):
            # train the model
            train_loss = 0
            model.train()
            for step, (seq, label) in enumerate(train_data_loader):  # the label here is the output vector
                # Forward
                # print(seq.clone().detach())
                seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
                output = model(seq)  # The output corresponds to the output corresponding to each batch size input
                loss = criterion(output, label.to(device))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss_list.append(train_loss)
            # validate the model
            model.eval()
            for step, (seq, label) in enumerate(validation_data_loader):
                seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                loss_list.append(loss.item())
            print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_data_loader.dataset)))
            writer.add_scalar('train_loss', train_loss / len(train_data_loader.dataset), epoch + 1)
            # save every 100 epoch
            if (epoch + 1) % 100 == 0:
                save_path = model_output_directory + str(i + 1)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), save_path + '/' + str(i+1) + '_epoch=' + str(epoch+1)+ '.pt')
        # torch.save(model,str(i+1)+'.pkl')
        writer.close()
        print('Training finished')





