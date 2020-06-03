#!/usr/bin/python
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def generate_seq_label(file_path,num_of_layers,window_length):
    num_sessions = 0
    inputs = []
    outputs = []
    flag = 0
    with open(file_path, 'r') as f:
        x = f.readlines()
        for line in x:
            line=line.strip('\n')
            vectors = []
            if(line=='-1'):
                l=1
                break
            num_sessions += 1

            key_values=line.split(' ')
            for key_value in key_values:
                key_value=key_value.split(',')
                #将字符串转为数字
                for k1 in range(len(key_value)):
                            if(key_value[k1]!=''):
                                key_value[k1]=float(key_value[k1])  
                vectors.append(key_value)          
            # line = tuple(map(lambda n: n, map(float, line.strip().split())))
            
            
            #补全
            if(len(vectors)<window_length+1):
                for i in range(window_length-len(vectors)+1):
                    vectors.append([0.0]*10)
            for i in range(len(vectors) - window_length):
                    inputs.append(vectors[i: i + window_length])
                    outputs.append(vectors[i + window_length])

            # For each log key's log parameter value vector，the length of these vector is same，the meaning of value at each position is same
            # eg: [log_vector1, log_vector2, log_vector3] --> log_vector4
            # so each element of inputs is a sequence，and each element of that sequence is a sequence too
            # nn's output is the prediction of parameter value vector
        
        # if len(x) < 2*num_of_layers:
        #     flag = 1

    # for i in range(len(vectors) - window_length):
    #     inputs.append(vectors[i: i + window_length])
    #     outputs.append(vectors[i + window_length])
    # print(inputs)
    # print(inputs[0])
    data_set = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))

    if len(vectors) > 0 and flag==0:
        return data_set, len(vectors[0])
    else:
        return None, 0


def train_model2(model_dir,log_preprocessor_dir,num_epochs,batch_size,window_length,num_of_layers,learning_rate,hidden_size):
    log_value_folder = log_preprocessor_dir + 'logvalue_train/'
    model_output_directory = model_dir + 'model2/'
    log_template = 'model2_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
    file_names = os.listdir(log_value_folder)
    for i in range(len(file_names)):
        print(i)
        file_name = str(i+1)
        train_data_set_name = log_value_folder + file_name
        validation_data_set_name = train_data_set_name

        train_data_set, input_size = generate_seq_label(train_data_set_name,num_of_layers,window_length)
        if input_size == 0:
            continue
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        validation_data_set, _ = generate_seq_label(validation_data_set_name,num_of_layers,window_length)
        validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        out_size = input_size
        model = Model(input_size, hidden_size, num_of_layers, out_size).to(device)
        
        writer = SummaryWriter(logdir =model_dir + 'log2/' + str(i + 1) + '_' + log_template)

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
            # save every 50 epoch
            if (epoch + 1) % 50 == 0:
                save_path = model_output_directory + str(i + 1)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), save_path + '/' + str(i+1) + '_epoch=' + str(epoch+1)+ '.pt')
        # torch.save(model,str(i+1)+'.pkl')
        writer.close()
        print('Training finished')





