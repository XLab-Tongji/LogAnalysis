# -*- coding: UTF-8 -*-
import torch
import os
import torch.nn as nn
import time
from anomalydetection.loganomaly.log_anomaly_sequential_train import Model

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# len(line) < window_length

def generate(name, window_length):
    log_keys_sequences = list()
    with open(name, 'r') as f:
        for line in f.readlines():
            line = tuple(map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            log_keys_sequences.append(tuple(line))
    return log_keys_sequences



def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path):

    model1 = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, test_file_path, num_candidates, pattern_vec_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    abnormal_loader = generate(test_file_path, window_length)
    abnormal_label = []
    with open(anomaly_test_line_path) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in abnormal_loader:
            i = 0
            # first traverse [0, window_size)
            while i < len(line) - window_length:
                lineNum = current_file_line * 10 + i + window_length + 1
                count_num += 1
                seq = line[i:i + window_length]
                label = line[i + window_length]
                print(label)
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                print(seq.shape)
                #label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                print(output)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                print('{} - predict result: {}, true label: {}'.format(count_num, predicted, vec_to_class_type[tuple(label)]))
                if lineNum in abnormal_label:  ## 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                else:
                    i += 1
                ALL += 1
                if vec_to_class_type[tuple(label)] not in predicted:
                    if lineNum in abnormal_label:
                        TN += 1
                    else:
                        FN += 1
                else:
                    if lineNum in abnormal_label:
                        FP += 1
                    else:
                        TP += 1
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

    #draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'], [Acc, P, R, F1], 'evaluations', '%')