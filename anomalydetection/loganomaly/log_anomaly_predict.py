import torch
import os
import torch.nn as nn
import time
import numpy as np
from anomalydetection.loganomaly.log_anomaly_train import Model
from anomalydetection.loganomaly.log_anomaly_train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_test_label(logkey_path, window_length,num_of_classes):
    f = open(logkey_path,'r')
    keys = f.readline().split()
    keys = list(map(int, keys))
    print(keys)
    length = len(keys)
    input_1 = np.zeros((length -window_length,num_of_classes))
    output_1 = np.zeros(length -window_length,dtype=np.int)
    input_2 = np.zeros((length -window_length,num_of_classes))
    output = np.zeros(length -window_length,dtype=np.int)
    for i in range(0,length -window_length):
        for t in range(0,num_of_classes):
            input_1[i][t] = keys[i]
        for j in range(i,i+window_length):
            input_2[i][keys[j]-1] += 1
        output[i] = keys[i+window_length]-1
    new_input_1 = np.zeros((length -2*window_length+1,window_length,num_of_classes))
    new_input_2 = np.zeros((length - 2 * window_length + 1, window_length, num_of_classes))
    for i in range(0,length -2*window_length+1):
        for j in range(i,i+window_length):
            new_input_1[i][j - i] = input_1[j]
            new_input_2[i][j-i] = input_2[j]
    new_output = output[window_length-1:]
    return length,new_input_1,new_input_2,new_output

def load_model(input_size_1,input_size_2, hidden_size, num_layers, num_classes, model_path):
    model = Model(input_size_1,input_size_2,hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print('model_path: {}'.format(model_path))
    return model

def filter_small_top_k(predicted, output):
    filter = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter.append(p)
    return filter

def do_predict(input_size_1,input_size_2, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, num_candidates, logkey_path):
    model = load_model(input_size_1,input_size_2 ,hidden_size, num_layers, num_classes, model_path)
    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    length,input_1,input_2,output = generate_test_label(logkey_path, window_length,num_classes)
    abnormal_label = []
    with open(anomaly_test_line_path) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for i in range(0,length-2*window_length+1):
            lineNum = i + 2*window_length
            seq = input_1[i]
            quan = input_2[i]
            label = output[i]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size_1).to(device)
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size_2).to(device)
            test_output = model(seq,quan)
            predicted = torch.argsort(test_output , 1)[0][-num_candidates:]
            predicted = filter_small_top_k(predicted, test_output)
            print('{} - predict result: {}, true label: {}'.format(lineNum, predicted,label))
            if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                i += 2*window_length + 1
            else:
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

if __name__=='__main__':
    input_size_1 = 61
    input_size_2 = 61
    hidden_size = 30
    num_of_layers = 2
    num_of_classes = 61
    num_epochs = 100
    batch_size = 200
    window_length = 5
    train_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_train'
    test_logkey_path = '../../Data/FTTreeResult-HDFS/deeplog_files/logkey/logkey_test'
    train_root_path = '../../Data/FTTreeResult-HDFS/model_train/'
    label_file_name = '../../Data/FTTreeResult-HDFS/deeplog_files/HDFS_abnormal_label.txt'
    model_out_path = train_root_path + 'model_out/'

    do_predict(input_size_1,input_size_2, hidden_size, num_of_layers, num_of_classes, window_length,
               model_out_path + 'Adam_batch_size=200;epoch=100.pt', label_file_name, 5, test_logkey_path)