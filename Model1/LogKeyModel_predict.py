#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import time
import argparse
from . import *


# Device configuration
device = torch.device("cpu")
# Hyperparameters，注意这里的window_size, input_size, hidden_size, num_layers, num_classes同train时的参数设置一致
window_size = 6
input_size = 1
hidden_size = 20
num_layers = 3
num_classes = 50
num_candidates = 3
model_dir='output/model'
model_path = model_dir + '/Adam_batch_size=200;epoch=100.pt'


def generate(name):
    # If you what to replicate the DeepLog paper clusters(Actually, I have a better result than DeepLog paper clusters),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    logkeys_sequences = set()
    # hdfs = []
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_size + 1 - len(line))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            logkeys_sequences.add(tuple(line))
    print('Number of sessions({}): {}'.format(name, len(logkeys_sequences)))
    return logkeys_sequences


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=num_layers, type=int)
    parser.add_argument('-hidden_size', default=hidden_size, type=int)
    parser.add_argument('-window_size', default=window_size, type=int)
    parser.add_argument('-num_candidates', default=num_candidates, type=int)
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    num_candidates = args.num_candidates

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))
    test_file_name = '../Data/Vectors1/SYSLOG_293_LogKeys_test'
    # test_loader = generate(test_file_name)
    test_normal_loader = generate(test_file_name)
    # test_abnormal_loader = generate('hdfs_test_abnormal')
    TP = 0
    FP = 0
    ALL=0
    # Test the model
    start_time = time.time()
    print('test normal data:')
    with torch.no_grad():
        count_num = 0
        for line in test_normal_loader:
            for i in range(len(line) - window_size):
                count_num += 1
                _seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(_seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                ALL+=1
                if label not in predicted:
                    print('{} - seq: {}, predict result: {}, true label: {}'.format(count_num, _seq, predicted, label))
                    FP += 1
                    # break
    # print('test abnormal data:')
    # with torch.no_grad():
    #     count_num = 0
    #     for line in test_abnormal_loader:
    #         for i in range(len(line) - window_size):
    #             count_num += 1
    #             seq = line[i:i + window_size]
    #             label = line[i + window_size]
    #             seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
    #             label = torch.tensor(label).view(-1).to(device)
    #             output = model(seq)
    #             predicted = torch.argsort(output, 1)[0][-num_candidates:]
    #             print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
    #             if label not in predicted:
    #                 TP += 1
    #                 break

    # Compute precision, recall and F1-measure
    # FN = len(test_abnormal_loader) - TP
    # P = 100 * TP / (TP + FP)
    # R = 100 * TP / (TP + FN)
    # F1 = 2 * P * R / (P + R)
    # print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
    print('false positive(FP): {}'.format(FP/ALL))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

