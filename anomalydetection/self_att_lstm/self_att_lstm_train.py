# -*- coding: UTF-8 -*-
# regularization waiting for heliren sparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# use cuda if available  otherwise use cpu
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, out_size, if_bidirectional, sequen_len):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True, bidirectional=if_bidirectional, dropout=0.5)
        if if_bidirectional:
            self.num_of_directions = 2
        else:
            self.num_of_directions = 1
        self.fc = nn.Linear(hidden_size*self.num_of_directions, out_size)

        self.att_weight = nn.Parameter(torch.randn(1, 1, self.hidden_size*self.num_of_directions))
        self.att_bias = nn.Parameter(torch.randn(1, 1, sequen_len))

        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

# l1 regularization will add later
    def attention_net(self, H):
        # print(H.size()) = [batch, numdirec*hidden, seqlen]
        a = F.softmax(torch.matmul(self.att_weight, H) + self.att_bias, 2)
        a = torch.transpose(a, 1, 2)
        return torch.bmm(H, a)

    def init_hidden(self, size):
        # size self.batch_size same
        h0 = torch.zeros(self.num_of_layers*self.num_of_directions, size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers*self.num_of_directions, size, self.hidden_size).to(device)
        return (h0, c0)

    def forward(self, input):
        # h_n: hidden state h of last time step
        # c_n: hidden state c of last time step
        out, _ = self.lstm(input, self.init_hidden(input.size(0)))

        # out = torch.transpose(out, 0, 1)
        # out shape [batch, seqlen, numdirec*hidden]
        out = torch.transpose(out, 1, 2)
        # out shape [batch, numdirec*hidden, seqlen]
        att_out = self.attention_net(out)
        # att_out shape[batch, num_direc*hidden_size, 1]
        # att_out[:, :, 0] shape[batch, num_direc*hidden_size]
        out = self.fc(att_out[:, :, 0])
        # out shape[batch, num_of_class]
        return out


def generate_seq_label(file_path, window_length, pattern_vec_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1
    num_of_sessions = 0
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            num_of_sessions += 1
            line = tuple(map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            if len(line) < window_length + 1:
                continue
            for i in range(len(line) - window_length):
                input_data.append(line[i:i + window_length])
                # line[i] is a list need to read file form a dic{vec:log_key} to get log key
                output_data.append(vec_to_class_type[line[i + window_length]])
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set

def generate_logdeep_seq_label(file_path, window_length):
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = tuple(map(lambda n: n-1, map(int, line.strip().split())))
            if len(line) < window_length + 1:
                continue
            for i in range(len(line) - window_length):
                input_data.append(line[i:i + window_length])
                # line[i] is a list need to read file form a dic{vec:log_key} to get log key
                output_data.append(line[i + window_length])
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set


def train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory, data_file, pattern_vec_file):
    # log setting
    log_directory = root_path + 'log_out/'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

    print("Train num_classes: ", num_of_classes)
    model = Model(input_size, hidden_size, num_of_layers, num_of_classes, False, window_length).to(device)
    # create data set
    sequence_data_set = generate_seq_label(data_file, window_length, pattern_vec_file)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)
    writer = SummaryWriter(logdir=log_directory + log_template)

    # Loss and optimizer  classify job
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
            output = model(seq)

            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')