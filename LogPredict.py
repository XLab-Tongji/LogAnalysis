#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import time
from enum import Enum
from Model1.LogKeyModel_train import Model as Model1
from Model2.KeyLogModel_train import Model as Model2
import argparse
import torch.nn as nn
import os

# Device configuration
device = torch.device("cpu")

# relation between log_pattern log_key log_line
pattern2log = []
pattern_dic = {}
# 存所有 log 的 value vector，为二维数组，第一维是 log_key 的编号，第二维该 log_key 第几条日志，
# 例如 pattern2value[1][2] 代表第一个 log_key 的第二条日志的 value vector
pattern2value = []
window_size = 4
input_size1 = 1
num_candidates = 3

RootPath = "./Data/LogClusterResult-k8s/"
log_address = "./k8s-2/k8s-2_abnormal.log"
abnormal_label_file = './k8s-2/k8s_label.txt'
mse_threshold = 0.1


# 继承枚举类
class LineNumber(Enum):
    PATTERN_LINE = 0
    NUMBERS_LINE = 3


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


def value_log_cluster():
    log_value_folder_cluster = RootPath + 'logvalue/logvalue_abnormal/'
    file_names = os.listdir(log_value_folder_cluster)
    pattern_key = 0
    pattern2value.append([])
    for i in range(len(file_names)):
        pattern2value.append([])
        with open(log_value_folder_cluster + str(i+1) + ".log", 'r') as in_text:
            for line in in_text.readlines():
                line = list(map(lambda n: n, map(float, line.strip().split())))
                pattern2value[i+1].append(line)


def parse_log_cluster():
    log_pattern_folder_cluster = RootPath + 'clusters/'
    file_names = os.listdir(log_pattern_folder_cluster)
    pattern_key = 0
    for i in range(len(file_names)):
        with open(log_pattern_folder_cluster + str(i+1) + ".log", 'r') as in_text:
            num_of_line = 0
            pattern = ''
            log_set = set()
            for line in in_text.readlines():
                if num_of_line == LineNumber.PATTERN_LINE.value:
                    pattern = line
                    num_of_line = num_of_line + 1
                elif num_of_line == LineNumber.NUMBERS_LINE.value:
                    lineNumbers = line.strip().split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = pattern
                    pattern_key = pattern_key + 1
                else:
                    num_of_line = num_of_line + 1


def parse_sequencer():
    if_first = True
    log_pattern_address_sequencer = ''
    with open(log_pattern_address_sequencer, 'rb') as in_text:
        log_set = set()
        pattern_key = 0
        last_pattern = ''
        for line in in_text.readlines():
            if (not line.startswith('#'.encode(encoding='utf-8'))) and len(line.strip()):
                if line.startswith('%msgtime%'.encode(encoding='utf-8')):
                    if if_first:
                        last_pattern = line
                        if_first = False
                        continue
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = last_pattern
                    pattern_key = pattern_key + 1
                    log_set = set()
                    last_pattern = line
                else:
                    line = line.decode(encoding='utf-8', errors='strict').strip()
                    lineNumbers = line.split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
    pattern2log.append(log_set)
    pattern_dic[pattern_key] = last_pattern


def loadModel1():
    hidden_size = 20
    num_layers = 3
    num_classes = len(pattern2value) + 1
    print("Model1 num_classes: ", num_classes)
    model_dir = RootPath + 'output/model1/'
    model_path = model_dir + 'Adam_batch_size=200;epoch=1000.pt'
    model1 = Model1(input_size1, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def loadModel2():
    model2_dir = RootPath + 'output/model2/'
    model2 = []
    for i in range(len(pattern2value)):
        hidden_size = 20  # 64
        num_layers = 3  # 2
        if len(pattern2value[i]) == 0:
            model2.append(None)
            continue
        input_size = len(pattern2value[i][0])
        out_size = input_size
        model_name = str(i+1) + '_epoch=300.pt'
        model_path = model2_dir + str(i+1) + '/' + model_name
        if not os.path.exists(model_path):
            model2.append(None)
            continue
        model = Model2(input_size, hidden_size, num_layers, out_size).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print('model_path: {}'.format(model_path))
        model2.append(model)
    return model2


if __name__ == '__main__':
    # pattern_source = 1
    # if pattern_source == 0:
    #     parse_sequencer()
    # else:
    #     parse_log_cluster()
    value_log_cluster()
    model1 = loadModel1()
    model2 = loadModel2()
    
    # 用于 Model2 的预测。存储每个log_key 目前预测到了第几条日志。当模型一预测为正常时，模型二则会进行预测。
    # 此时需要待预测的 log_key 的前几条相同 log_key 的日志，故用 pattern_index 记录待预测的 log_key 是第几条。
    pattern_index = [0]*len(pattern2value)
    start_time = time.time()
    criterion = nn.MSELoss()  # 用于回归预测
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    abnormal_loader = generate(RootPath + 'logkey/logkey_abnormal')
    abnormal_label = []
    with open(abnormal_label_file) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        for line in abnormal_loader:
            i = 0
            # 先遍历 [0, window_size)
            for ii in range(window_size):
                if ii < len(line):
                    pattern_index[line[ii]] += 1
            while i < len(line) - window_size:
                lineNum = i + window_size + 1
                count_num += 1
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size1).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model1(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
                now_pattern_index = pattern_index[label]
                if lineNum in abnormal_label: ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    for j in range(window_size + 1):
                        if i + window_size + j < len(line) and line[i + window_size + j] < len(pattern_index):
                            pattern_index[line[i + window_size + j]] += 1
                        else:
                            break
                    i += window_size+1
                else:
                    pattern_index[label] += 1
                    i += 1
                ALL += 1
                if label not in predicted:
                    if lineNum in abnormal_label:
                        TN += 1
                    else:
                        FN += 1
                else:
                    # 模型一检测出来正常的，用模型二检测
                    # values：属于（编号为 pattern_id 的 log_key）的所有日志的 value vector
                    values = pattern2value[label]
                    vi = now_pattern_index
                    if vi >= window_size and vi < len(values):
                        # Model2 进行测试
                        seq2 = values[vi-window_size:vi]
                        label2 = values[vi]
                        seq2 = torch.tensor(seq2, dtype=torch.float).view(-1, window_size, len(seq2[0])).to(device)
                        label2 = torch.tensor(label2).view(-1).to(device)
                        mse = 0
                        if label < len(model2) and model2[label] != None:
                            output = model2[label](seq2)
                            # 计算预测结果和原始结果的MSE，若MSE在高斯分布的置信区间以内，则该日志是正常日志，否则为异常日志
                            # 此处用正常日志流做test，故只需要计算TP、FP
                            # predicted = torch.argsort()
                            mse = criterion(output[0], label2.to(device))
                        
                        if mse < mse_threshold:
                            if lineNum in abnormal_label:
                                FP += 1
                            else:
                                TP += 1
                        else:
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
    if TP + FP == 0:
        P = 0
    else:
        P = 100 * TP / (TP + FP)

    if TP + FN == 0:
        R = 0
    else:
        R = 100 * TP / (TP + FN)

    if P + R == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)

    Acc = (TP + TN) * 100 / ALL

    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

