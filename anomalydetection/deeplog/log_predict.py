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
from collections import Counter

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 记录每个 key 对应的 value 的长度
value_length_of_key = []

# 继承枚举类
class LineNumber(Enum):
    PATTERN_LINE = 0
    NUMBERS_LINE = 3



def generate(name,window_length):
    log_keys_sequences=list()
    length=0
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_length + 1 - len(line))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            # log_keys_sequences[tuple(line)] = log_keys_sequences.get(tuple(line), 0) + 1
            log_keys_sequences.append(tuple(line))
            length+=1
    return log_keys_sequences,length


def get_value_length(log_preprocessor_dir,log_fttree_out_dir):
    global value_length_of_key
    value_length_of_key = [10]*(len(os.listdir(log_fttree_out_dir)) + 1)
    log_value_folder = log_preprocessor_dir + '/train/logvalue/normal/'
    file_names = os.listdir(log_value_folder)
    # for i in range(len(file_names)):
    #     with open(log_value_folder + str(i+1), 'r') as f:
    #         x = f.readlines()
    #         if len(x) == 0 or x[0].strip('\n') == '-1':
    #             value_length_of_key.append(0)
    #         else:
    #             line = x[0].strip('\n')
    #             key_values = line.split(' ')
    #             value_length_of_key[i+1] = len(key_values[0].split(','))


def load_model1(model_dir,model_name,input_size, hidden_size, num_layers):
    num_classes = len(value_length_of_key)
    # num_classes = 28
    print("Model1 num_classes: ", num_classes)
    model1_dir = model_dir + 'model1/'
    model_path = model1_dir + model_name
    model1 = Model1(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def load_model2(model_dir,epoch,input_size, hidden_size, num_layers):
    model2_dir = model_dir+ 'model2/'
    model2 = []
    for i in range(len(value_length_of_key)):
        if value_length_of_key[i] == 0:
            model2.append(None)
            continue
        input_size = value_length_of_key[i]
        out_size = input_size
        model_name = str(i+1) + '_epoch=' + str(epoch)+ '.pt'
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


def do_predict(log_preprocessor_dir,log_fttree_out_dir,model_dir,model1_name,model2_num_epochs,window_length,input_size, hidden_size, num_layers,num_candidates,mse_threshold,use_model2):
    # abnormal_label_file = log_preprocessor_dir + 'HDFS_abnormal_label.txt'

    get_value_length(log_preprocessor_dir,log_fttree_out_dir)

    model1 = load_model1(model_dir, model1_name, input_size, hidden_size, num_layers)
    
    model2 = load_model2(model_dir,model2_num_epochs,10, hidden_size, num_layers)

    # for Model2's prediction, store which log currently predicts for each log_key.
    # When model one predicts normal, model2 makes predictions.
    # At this time, the forward few logs with the same log_key are needed to be predicted
    # so the pattern_index is used to record the log_key to be predicted.
    #pattern_index = [0]*len(pattern2value)
    #pattern_index = [0] * 63
    start_time = time.time()
    criterion = nn.MSELoss()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    test_normal_loader, test_normal_length  = generate(log_preprocessor_dir+ '/test/logkey/normal',window_length)
    test_abnormal_loader, test_abnormal_length=generate(log_preprocessor_dir+'/test/logkey/abnormal',window_length)
    

    print('predict start')
    
    #normal test
    with torch.no_grad():
        count = 1
        for line_num,line in enumerate(test_normal_loader):
            model1_success=False
            for i in range(len(line) - window_length-1):
                seq0 = line[i:i + window_length]
                label = line[i + window_length]
               

                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1,window_length,input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model1(seq0)
                predicted = torch.argsort(output,
                                            1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    model1_success=True
                    break 
            if(model1_success):
                continue

            
            #如果模型二预测normal   TN+1  否则FP+1

            #现在有63个预测normal value 文件  对一个line  找对应的 value normal下的行 进行预测   
            
            # When model one predicts normal, model2 makes predictions.
            # values：all log's value vector belongs to log_key（whose id is pattern_id）
            # 是否使用模型二
            if use_model2:

                seq=[]  #得到63个normal预测文件下的这个window的seq
                for i in range(31):
                    with open(log_preprocessor_dir+'/test/logvalue/normal/'+str(i+1),'r')as f:
                        key_values=f.readlines()
                        key_values=key_values[line_num].strip('\n')
                        if(key_values=='-1'):
                            continue
                        seq.append(key_values.split(' '))
                #将字符串转为数字
                for k1 in range(len(seq)):
                    for k2 in range(len(seq[k1])):
                        seq[k1][k2]=seq[k1][k2].strip('\n')
                        seq[k1][k2]=seq[k1][k2].split(',')
                        for k3 in range(len(seq[k1][k2])):
                            if(seq[k1][k2][k3]!=''):
                                seq[k1][k2][k3]=float(seq[k1][k2][k3])
                
                #补全
                for i in range(len(seq)):
                    if(len(seq[i])<window_length+1):
                        for j in range(window_length+1- len(seq[i])):
                            seq[i].append([0.0]*10) 
                model2_success=False
                #预测
                for i in range(len(seq)):
                    if(model2[i]==None):
                        continue
                    for j in range(len(seq[i]) - window_length):
                        seq2 =seq[i][j:j + window_length]
                        label2= seq[i][j + window_length]

                        seq2 = torch.tensor(seq2, dtype=torch.float).view(
                            -1,window_length,10).to(device)
                        label2 = torch.tensor(label,dtype=torch.float).view(-1).to(device)
                        output = model2[i](seq2)
                        mse = criterion(output[0], label2.to(device))
                        if mse > mse_threshold:
                            FP+=1
                            model2_success=True
                            break
                    if(model2_success):
                        break

    
    #abnormal test
    with torch.no_grad():
        for line_num,line in enumerate(test_abnormal_loader):
            model1_success=False
            for i in range(len(line) - window_length):
                seq0 = line[i:i + window_length]
                label = line[i + window_length]

                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1, window_length, input_size).to(device)
                
                label = torch.tensor(label,).view(-1).to(device)
                output = model1(seq0)
                predicted = torch.argsort(output,
                                            1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    model1_success=True
                    break
            if(model1_success):
                continue

        # 是否使用模型二
        if use_model2:
            seq=[]  #得到63个normal预测文件下的这个window的seq
            for i in range(31):
                with open(log_preprocessor_dir+'/test/logvalue/abnormal/'+str(i+1),'r')as f:
                    key_values=f.readlines()
                    key_values=key_values[line_num].strip('\n')
                    if(key_values=='-1'):
                        continue
                    seq.append(key_values.split(' '))
            #将字符串转为数字
            for k1 in range(len(seq)):
                for k2 in range(len(seq[k1])):
                    seq[k1][k2]=seq[k1][k2].strip('\n')
                    seq[k1][k2]=seq[k1][k2].split(',')
                    for k3 in range(len(seq[k1][k2])):
                        if(seq[k1][k2][k3]!=''):
                            seq[k1][k2][k3]=float(seq[k1][k2][k3])
            
            #补全
            for i in range(len(seq)):
                if(len(seq[i])<window_length+1):
                    for j in range(window_length+1- len(seq[i])):
                        seq[i].append([0.0]*10)
            # 预测
            model2_success = False
            for i in range(len(seq)):
                if (model2[i] == None):
                    continue
                for j in range(len(seq[i]) - window_length):
                    seq2 = seq[i][j:j + window_length]
                    label2 = seq[i][j + window_length]

                    seq2 = torch.tensor(seq2, dtype=torch.float).view(
                        -1, window_length, 10).to(device)
                    label2 = torch.tensor(label, dtype=torch.float).view(-1).to(device)
                    output = model2[i](seq2)
                    mse = criterion(output[0], label2)
                    if mse > mse_threshold:
                        TP += 1
                        model2_success = True
                        break
                if (model2_success):
                    break

        #现在有63个预测normal value 文件  对一个line  找对应的 value normal下的行 进行预测   


    # Compute precision, recall and F1-measure
    FN = test_abnormal_length - TP
    TN=test_normal_length-FP
    
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    Acc = (TP + TN) * 100 /(TP+TN+FP+FN)
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))








