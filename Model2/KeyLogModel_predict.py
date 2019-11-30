#!/usr/bin/python
# -*-coding:utf-8-*-

import torch
import torch.nn as nn
import time
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
window_size = 10  # 10
hidden_size = 20  # 64
num_layers = 3  # 2
# num_classes = 2  # 28
# input_size = 2  #  x的特征维度。输入数据的维度，对于model2来说，长度为每个key对应的log vector的数据长度
# out_size = 2
num_epochs = 5  # 300
batch_size = 20  # 2048
num_candidates = 9

RootPath = '../Data/LogClusterResult-5G/'
log_value_folder = RootPath + 'logvalue/logvalue_test/'

mse_threshold = 0.1


def generate(name):
    vectors = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(float, line.strip().split())))
            vectors.append(tuple(line))
    return vectors, len(vectors[0])


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)
        # self.out = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, (h_n, c_n) = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])  # 最后一个时步的输出
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
    for ii in range(len(file_names)):
        modelfile = RootPath + 'output/model2/' + str(ii+1) + '/'
        if not os.path.exists(modelfile):
            print(modelfile + " doesn't exist")
            print("class " + str(ii+1) + " doesn't exist in train data")
            continue
        for model_path in os.listdir(modelfile):
            test_loader, input_size = generate(log_value_folder + str(ii+1) + '.txt')
            out_size = input_size
            model = Model(input_size, hidden_size, num_layers, out_size).to(device)
            model.load_state_dict(torch.load(modelfile + model_path))
            model.eval()
            print('model_path: {}'.format(model_path))
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            # test
            # 此处的数据应包含正常日志和异常日志，且分开在两个数据集中
            criterion = nn.MSELoss()  # 用于回归预测
            # 对模型的评估
            start_time = time.time()
            with torch.no_grad():
                # 对正常数据集的测试
                for i in range(len(test_loader) - window_size):
                    seq = test_loader[i:i+window_size]
                    label = test_loader[i+window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    # 计算预测结果和原始结果的MSE，若MSE在高斯分布的置信区间以内，则该日志是正常日志，否则为异常日志
                    # 此处用正常日志流做test，故只需要计算TP、FP
                    # predicted = torch.argsort()
                    # print(output)
                    mse = criterion(output[0], label.to(device))
                    # print(mse)
                    if mse < mse_threshold:
                        TP += 1
                    else:
                        FP += 1
                # # 对异常数据集的测试
                P = 100.0 * float(TP) / float(TP + FP)  # precision
                R = 0  # Recall
                F1 = 0  # F1
                print(
                    'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
                        FP, FN, P, R, F1))
                print('Finished Predicting')
                elapsed_time = time.time() - start_time
                print('elapsed_time: {}'.format(elapsed_time))
