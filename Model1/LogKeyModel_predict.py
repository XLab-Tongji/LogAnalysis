#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import time
import os
import argparse
from . import *


# Device configuration
device = torch.device("cpu")
# Hyperparameters，注意这里的window_size, input_size, hidden_size, num_layers, num_classes同train时的参数设置一致
window_size = 6
input_size = 1
hidden_size = 20
num_layers = 3
num_classes = 9
num_candidates = 3
RootPath = '../Data/LogClusterResult-k8s/'
abnormal_label_file = '../k8s-2/k8s_label.txt'
model_dir = RootPath + 'output/model1/'
test_file_name = RootPath + 'logkey/logkey_test'
abnormal_file_name = RootPath + 'logkey/logkey_abnormal'

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

def generate_abnomal_label(name):
    with open(name, 'r') as f:
        line = f.readline()
        line = list(map(lambda n: n, map(int, line.strip().split())))
    return line

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
    num_classes = len(os.listdir(RootPath + 'clusters/')) + 2
    outfile = open(RootPath + 'output/reslut_test_model1.txt', 'w')
    abnormal_label = generate_abnomal_label(abnormal_label_file)
    for i in range(10):
        model_path = model_dir + '/Adam_batch_size=200;epoch=' + str((i+1)*100) + '.pt'
        model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('model_path: {}'.format(model_path))
        # test_loader = generate(test_file_name)
        test_normal_loader = generate(test_file_name)
        test_abnormal_loader = generate(abnormal_file_name)
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        ALL = 0
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
                    ALL += 1
                    if label not in predicted:
                        print('{} - seq: {}, predict result: {}, true label: {}'.format(count_num, _seq, predicted, label))
                        FN += 1
                    
                        # break
        TP += ALL - FN
        print('test abnormal data:')
        
        with torch.no_grad():
            count_num = 0
            for line in test_abnormal_loader:
                i = 0
                while i < len(line) - window_size:
                    lineNum = i + window_size + 1
                    count_num += 1
                    seq = line[i:i + window_size]
                    label = line[i + window_size]
                    seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                    label = torch.tensor(label).view(-1).to(device)
                    output = model(seq)
                    predicted = torch.argsort(output, 1)[0][-num_candidates:]
                    print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
                    if lineNum in abnormal_label: ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                        i += window_size+1
                    else:
                        i += 1
                    ALL += 1
                    if label not in predicted:
                        if lineNum in abnormal_label:
                            TN += 1
                        else:
                            FN += 1
                    else:
                        if lineNum in abnormal_label:
                            FP += 1
                        else:
                            TP += 1

        # Compute precision, recall and F1-measure
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        Acc=(TP+TN)*100/ALL
        print(model_path,file=outfile)
        print('true positive (TP): {},false positive (FP): {}, true Negative (TN): {},false negative (FN): {}'.format(TP, FP,TN,FN),file=outfile)
        print('Acc: {:.3f}% ,Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc,P, R, F1),file=outfile)
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        print('',file=outfile)

