import torch
import os
import torch.nn as nn
import time
import numpy as np
from anomalydetection.loganomaly.log_anomaly_quantitive_train import Model
from anomalydetection.loganomaly.log_anomaly_quantitive_train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_test_label(logkey_path, window_length):
    f = open(logkey_path,'r')
    keys = f.readline().split()
    keys = list(map(int, keys))
    print(keys)
    length = len(keys)
    input = np.zeros((length -window_length,num_of_classes))
    output = np.zeros(length -window_length,dtype=np.int)
    for i in range(0,length -window_length):
        for j in range(i,i+window_length):
            input[i][keys[j]-1] += 1
        output[i] = keys[i+window_length]-1
    new_input = np.zeros((length -2*window_length+1,window_length,num_of_classes))
    for i in range(0,length -2*window_length+1):
        for j in range(i,i+window_length):
            new_input[i][j-i] = input[j]
    new_output = output[window_length-1:]
    print(new_input.shape)
    print(new_output.shape)
    print(new_input[0])
    print(new_output[0])
    return length,new_input,new_output

def load_quantitive_model(input_size, hidden_size, num_layers, num_classes, model_path):
    model2 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model2.load_state_dict(torch.load(model_path, map_location='cpu'))
    model2.eval()
    print('model_path: {}'.format(model_path))
    return model2

def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, num_candidates, logkey_path):
    quantitive_model = load_quantitive_model(input_size, hidden_size, num_layers, num_classes, model_path)
    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    length,input,output = generate_test_label(logkey_path, window_length)
    abnormal_label = []
    with open(anomaly_test_line_path) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for i in range(0,length-2*window_length+1):
            lineNum = i + 2*window_length
            quan = input[i]
            label = output[i]
            quan = torch.tensor(quan, dtype=torch.float).view(-1, window_length, input_size).to(device)
            test_output = quantitive_model(quan)
            predicted = torch.argsort(test_output , 1)[0][-num_candidates:]
            print('{} - predict result: {}, true label: {}'.format(lineNum, predicted,label))
            if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                i += 2*window_length + 1
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

input_size = 61
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
model_out_path = train_root_path + 'quantitive_model_out/'

# train_model(window_length, input_size, hidden_size,
#             num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path,
#             model_out_path,train_logkey_path)

do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length,
           model_out_path + 'Adam_batch_size=200;epoch=100.pt', label_file_name, 3, test_logkey_path)

