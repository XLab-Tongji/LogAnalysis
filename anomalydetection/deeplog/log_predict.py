#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import time
from enum import Enum
from anomalydetection.deeplog.Model1.log_key_LSTM_train import Model as Model1
from anomalydetection.deeplog.Model2.variable_LSTM_train import Model as Model2
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pattern2value = []

# 继承枚举类
class LineNumber(Enum):
    PATTERN_LINE = 0
    NUMBERS_LINE = 3


def generate(name,window_length):
    log_keys_sequences = list()
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_length + 1 - len(line))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            log_keys_sequences.append(tuple(line))
    return log_keys_sequences


def value_log_cluster(log_preprocessor_dir):
    log_value_folder_cluster = log_preprocessor_dir + 'logvalue_test/'
    file_names = os.listdir(log_value_folder_cluster)
    pattern2value.append([])
    for i in range(len(file_names)):
        pattern2value.append([])
        with open(log_value_folder_cluster + str(i+1) + ".txt", 'r') as in_text:
            for line in in_text.readlines():
                line = list(map(lambda n: n, map(float, line.strip().split())))
                pattern2value[i+1].append(line)


def load_model1(model_dir,input_size, hidden_size, num_layers):
    #num_classes = len(pattern2value) + 1
    num_classes = 63
    print("Model1 num_classes: ", num_classes)
    model1_dir = model_dir + 'model1/'
    model_path = model1_dir + 'Adam_batch_size=200;epoch=100.pt'
    model1 = Model1(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def load_model2(model_dir,input_size, hidden_size, num_layers):
    model2_dir = model_dir+ 'model2/'
    model2 = []
    for i in range(len(pattern2value)):
        if len(pattern2value[i]) == 0:
            model2.append(None)
            continue
        input_size = len(pattern2value[i][0])
        out_size = input_size
        model_name = str(i+1) + '_epoch=50.pt'
        model_path = model2_dir + str(i+1) + '/' + model_name
        if not os.path.exists(model_path):
            model2.append(None)
            continue
        model = Model2(input_size, hidden_size, num_layers, out_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print('model_path: {}'.format(model_path))
        model2.append(model)
    return model2


def draw_evaluation(title, indexs, values, xlabel, ylabel):
    fig = plt.figure(figsize=(15,10))
    x = indexs
    y = values
    plt.bar(x, y, align='center', alpha=0.5, width=0.4)
    plt.xticks(x, x)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def do_predict(log_preprocessor_dir,model_dir,window_length,input_size, hidden_size, num_layers,num_candidates,mse_threshold):
    abnormal_label_file = log_preprocessor_dir + 'HDFS_abnormal_label.txt'

    #value_log_cluster(log_preprocessor_dir)
    model1 = load_model1(model_dir, input_size, hidden_size, num_layers)
    #model2 = load_model2(model_dir,input_size, hidden_size, num_layers)

    # for Model2's prediction, store which log currently predicts for each log_key.
    # When model one predicts normal, model2 makes predictions.
    # At this time, the forward few logs with the same log_key are needed to be predicted
    # so the pattern_index is used to record the log_key to be predicted.
    #pattern_index = [0]*len(pattern2value)
    pattern_index = [0] * 63
    start_time = time.time()
    criterion = nn.MSELoss()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    abnormal_loader = generate(log_preprocessor_dir+ 'logkey/logkey_test',window_length)
    abnormal_label = []
    with open(abnormal_label_file) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in abnormal_loader:
            i = 0
            # first traverse [0, window_size)
            for ii in range(window_length):
                if ii < len(line):
                    pattern_index[line[ii]] += 1
            while i < len(line) - window_length:
                lineNum = current_file_line * 10 + i + window_length + 1
                count_num += 1
                seq = line[i:i + window_length]
                for n in range(len(seq)):
                    if current_file_line * 10 + i + n + 1 in abnormal_label:
                        i = i + n + 1
                        continue
                label = line[i + window_length]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model1(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, label))
                now_pattern_index = pattern_index[label]
                if lineNum in abnormal_label: ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    for j in range(window_length + 1):
                        if i + window_length + j < len(line) and line[i + window_length + j] < len(pattern_index):
                            pattern_index[line[i + window_length + j]] += 1
                        else:
                            break
                    i += window_length + 1
                else:
                    pattern_index[label] += 1
                    i += 1
                ALL += 1
                if label not in predicted:
                    if lineNum in abnormal_label:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if lineNum in abnormal_label:
                        FN += 1
                    else:
                        TN += 1
                '''else:
                    # When model one predicts normal, model2 makes predictions.
                    # values：all log's value vector belongs to log_key（whose id is pattern_id）
                    values = pattern2value[label]
                    vi = now_pattern_index
                    if vi >= window_length and vi < len(values):
                        # Model2 testing
                        seq2 = values[vi - window_length:vi]
                        label2 = values[vi]
                        seq2 = torch.tensor(seq2, dtype=torch.float).view(-1, window_length, len(seq2[0])).to(device)
                        label2 = torch.tensor(label2).view(-1).to(device)
                        mse = 0
                        if label < len(model2) and model2[label] != None:
                            output = model2[label](seq2)
                            # Calculate the MSE of the prediction result and the original result.
                            # If the MSE is within the confidence interval of the Gaussian distribution, the log is a normal log
                            mse = criterion(output[0], label2.to(device))

                        if mse < mse_threshold:
                            print(mse, mse_threshold)
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
                    '''
            current_file_line += 1
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

    draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'],[Acc, P, R, F1], 'evaluations', '%')


