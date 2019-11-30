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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
window_size = 10  # 10
hidden_size = 20  # 64
num_layers = 3  # 2
# num_classes = 2  # 28
# input_size = 2  #  x的特征维度。输入数据的维度，对于model2来说，长度为每个key对应的log vector的数据长度
# out_size = 2
num_epochs = 300
batch_size = 20  # 2048
# in_features = 10
# out_features = in_features
learning_rate = 0.01
RootPath = '../Data/LogClusterResult-5G/'
log_value_folder = RootPath + 'logvalue/logvalue_train/'
model_dir = RootPath + 'output/model2/'
log = 'model2_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)

# files path info
# train_dataset_name = 'log_vectors1'
# validation_dataset_name = 'log_vectors1'


def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    vectors = []
    with open(name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n, map(float, line.strip().split())))
            vectors.append(line)
            # 对每个key的log parameter value vector来说，这些vector的长度一致，在某个位置的数字代表的意义一致
            # 预测示例如：[log_vector1, log_vector2, log_vector3] --> log_vector4
            # 故inputs中的每一项是一个序列，而序列中的每一项也是一个向量
            # 神经网络的output是一个预测的parameter value vector

            # for i in range(len(line) - window_size):
            #     inputs.append(line[i:i + window_size])
            #     outputs.append(line[i + window_size])
            # break
    for i in range(len(vectors) - window_size):
        inputs.append(vectors[i: i + window_size])
        outputs.append(vectors[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    # print(inputs)
    # print(inputs[0])
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    if len(vectors) > 0:
        return dataset, len(vectors[0])
    else:
        return None, 0


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)
        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        # 参考网址：https://www.jianshu.com/p/043083d114d4
        # h_n: 各个层的最后一个时步的隐含状态h;
        # c_n: 各个层的最后一个时步的隐含状态C
        out, (h_n, c_n) = self.lstm(input, (h0, c0))
        # print('out size:')
        # print(out.size())

        out = self.fc(out[:, -1, :])  # 最后一个时步的输出
        # print('out[:, -1, :]:')
        # print(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=3, type=int)  # default=2
    parser.add_argument('-hidden_size', default=20, type=int)  # default=64
    parser.add_argument('-window_size', default=2, type=int)  # default=10
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    file_names = os.listdir(log_value_folder)
    for i in range(len(file_names)):
        file_name = str(i+1) + ".txt"
        train_dataset_name = log_value_folder + file_name
        validation_dataset_name = train_dataset_name

        train_dataset, input_size = generate(train_dataset_name)
        if input_size == 0:
            continue
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        validation_dataset, _ = generate(validation_dataset_name)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        out_size = input_size
        model = Model(input_size, hidden_size, num_layers, out_size).to(device)
        
        writer = SummaryWriter(logdir = RootPath + '/output/log2/' + str(i+1) + '_' + log)

        # Loss and optimizer
        criterion = nn.MSELoss()  # 用于回归预测
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_list = []  # 用于记录训练过程中的loss值，这些值服从高斯分布
        train_loss_list = []  # 记录训练阶段的loss，之后作图用

        # Train and validate the model
        total_step = len(train_dataloader)
        for epoch in range(num_epochs):  # Loop over the dataset multiple times
            # train the model
            train_loss = 0
            model.train()
            for step, (seq, label) in enumerate(train_dataloader):  # the label here is the output vector
                # Forward pass
                # print(seq.clone().detach())
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq)  # 该output对应着每batch size个输入对应的输出
                loss = criterion(output, label.to(device))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss_list.append(train_loss)
            # validate the model
            model.eval()
            for step, (seq, label) in enumerate(validation_dataloader):
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                loss_list.append(loss.item())
            print('Epoch [{}/{}], Train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(train_dataloader.dataset)))
            writer.add_scalar('train_loss', train_loss / len(train_dataloader.dataset), epoch + 1)
            if (epoch+1) % 100 == 0:
                save_path = model_dir + str(i+1)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), save_path + '/' + str(i+1) + '_epoch=' + str(epoch+1)+ '.pt')
        # torch.save(model,str(i+1)+'.pkl')
        writer.close()
        print('Finished Training')

        # draw the Gaussian distribution of the loss in the validation
        num_bins = 100
        sigma = 1
        mu = 0
        fig, ax = plt.subplots()
        print(loss_list)
        # # the histogram of the data
        # n, bins, patches = ax.hist(loss_list, num_bins, density=True)
        # # add a "best fit" line
        # # y = mlab.normpdf(bins, mu, sigma)
        # # ax.plot(bins, y, '--')
        # ax.set_xlabel('loss value')
        # ax.set_ylabel('percentage')
        # ax.set_title('Gaussian distribution')
        # fig.tight_layout()
        # plt.show()

        # # 作训练迭代过程中的loss变化图
        # plt.plot(train_loss_list, label='loss for every epoch')
        # plt.legend()
        # plt.show()




