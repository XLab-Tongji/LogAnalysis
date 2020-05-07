import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def init_hidden(self, size):
        h0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers, size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, input):
        out, _ = self.lstm(input, self.init_hidden(input.size(0)))
        out = self.fc(out[:, -1, :])
        return out


def generate_quantitive_label(logkey_path, window_length,num_of_classes):
    f = open(logkey_path,'r')
    keys = f.readline().split()
    keys = list(map(int, keys))
    print(keys)
    length = len(keys)
    input = np.zeros((length -window_length,num_of_classes))
    output = np.zeros(length -window_length,dtype=np.int)
    for i in range(0,length -window_length):
        for j in range(i,i+window_length):
            input[i][keys[j]-1] += 1
        output[i] = keys[i+window_length]-1
    new_input = np.zeros((length -2*window_length+1,window_length,num_of_classes))
    for i in range(0,length -2*window_length+1):
        for j in range(i,i+window_length):
            new_input[i][j-i] = input[j]
    new_output = output[window_length-1:]
    dataset = TensorDataset(torch.tensor(new_input,dtype=torch.float),torch.tensor(new_output,dtype=torch.long))
    print(new_input.shape)
    print(new_output.shape)
    return dataset

def train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory,logkey_path):
    # log setting
    log_directory = root_path + 'quantitive_log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    model = Model(input_size, hidden_size, num_of_layers, num_of_classes).to(device)
    # create data set
    quantitive_data_set = generate_quantitive_label(logkey_path, window_length,num_of_classes)
    # create data_loader
    data_loader = DataLoader(dataset=quantitive_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (quan, label) in enumerate(data_loader):
            quan = quan.clone().detach().view(-1, window_length, input_size).to(device)
            output = model(quan)

            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % 100 == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')








