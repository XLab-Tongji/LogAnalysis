# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
import torch
import os
import torch.nn as nn
import time
from anomalydetection.self_att_lstm.self_att_lstm_train import Model
import torch.nn.functional as F

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

def generate_log_deep(name, window_length):
    log_keys_sequences = {}
    with open(name, 'r') as f:
        for line in f.readlines():
            if len(line) < window_length + 1:
                continue
            ln = list(map(lambda n: n-1, map(int, line.strip().split())))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            log_keys_sequences[tuple(ln)] = log_keys_sequences.get(tuple(ln), 0) + 1
    return log_keys_sequences


def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path, window_size):

    model1 = Model(input_size, hidden_size, num_layers, num_classes, if_bidirectional=False, sequen_len=window_size).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1

def filter_small_top_k(predicted, output):
    filter = []
    for p in predicted:
        if output[0][p] > 0.001:
            filter.append(p)
    return filter


def do_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, anomaly_test_line_path, test_file_path, num_candidates, pattern_vec_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path, window_length)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    abnormal_loader = generate(test_file_path, window_length)
    with open(anomaly_test_line_path) as f:
        abnormal_label = [int(x) for x in f.readline().strip().split()]
    # for testing model using train set
    # abnormal_label = []
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in abnormal_loader:
            i = 0
            # first traverse [0, window_size)
            while i < len(line) - window_length:
                lineNum = current_file_line * 200 + i + window_length + 1
                input_abnormal = False
                count_num += 1
                seq = line[i:i + window_length]
                origin_seq = seq
                label = line[i + window_length]
                for n in range(len(seq)):
                    if current_file_line * 200 + i + n + 1 in abnormal_label:
                        input_abnormal = True
                        continue
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                #label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                output = F.softmax(output, 1)
                # print(torch.sort(output, 1))
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                # print(predicted)
                # print('Fp {} - predict result: {}, true label: {}'.format(lineNum, predicted, vec_to_class_type[tuple(label)]))
                '''if lineNum in abnormal_label or in:  # 若出现异常日志，则接下来的预测跳过异常日志，保证进行预测的日志均为正常日志
                    i += window_length + 1
                else:
                    i += 1'''
                i += 1
                ALL += 1
                if vec_to_class_type[tuple(label)] not in predicted:
                    if lineNum in abnormal_label or input_abnormal:
                        TP += 1
                    else:
                        FP += 1

                else:
                    if lineNum in abnormal_label or input_abnormal:
                        print('FN {} - predict result: {}, true label: {}'.format(lineNum, predicted, vec_to_class_type[tuple(label)]))
                        print(torch.sort(output, 1))
                        for l in origin_seq:
                            print(str(vec_to_class_type[tuple(l)]), end='')
                            print(',', end='')
                        print(str(vec_to_class_type[tuple(label)]))
                        FN += 1
                    else:
                        TN += 1
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
    FAR = FP * 100 / (FP+TN)
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%, FAR: {:.3f}%'.format(Acc, P, R, F1, FAR))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

    #draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'], [Acc, P, R, F1], 'evaluations', '%')


def do_log_deep_predict(input_size, hidden_size, num_layers, num_classes, window_length, model_path, test_normal_file_path, test_abnormal_file_path, num_candidates, pattern_vec_file):

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path, window_length)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    ALL = 0
    normal_loader = generate_log_deep(test_normal_file_path, window_length)
    abnormal_loader = generate_log_deep(test_abnormal_file_path, window_length)
    # for testing model using train set
    # abnormal_label = []
    print('predict start')
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in normal_loader.keys():
            count_num += 1
            print(count_num)
            if count_num > 6000:
                break
            i = 0
            # first traverse [0, window_size)
            while i < len(line) - window_length:
                seq = line[i:i + window_length]
                label = line[i + window_length]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                #label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                output = F.softmax(output, 1)
                # print(torch.sort(output, 1))
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                # print(predicted)
                # print('Fp {} - predict result: {}, true label: {}'.format(lineNum, predicted, vec_to_class_type[tuple(label)]))
                if label in predicted:
                    TN += normal_loader[line]
                else:
                    FP += normal_loader[line]
                i += 1
    with torch.no_grad():
        count_num = 0
        current_file_line = 0
        for line in abnormal_loader.keys():
            count_num += 1
            i = 0
            # first traverse [0, window_size)
            while i < len(line) - window_length:
                seq = line[i:i + window_length]
                label = line[i + window_length]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                #label = torch.tensor(label).view(-1).to(device)
                output = sequential_model(seq)
                output = F.softmax(output, 1)
                # print(torch.sort(output, 1))
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                predicted = filter_small_top_k(predicted, output)
                # print(predicted)
                # print('Fp {} - predict result: {}, true label: {}'.format(lineNum, predicted, vec_to_class_type[tuple(label)]))
                if label in predicted:
                    FN += abnormal_loader[line]
                else:
                    TP += abnormal_loader[line]
                i += 1
            print(count_num)

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

    Acc = (TP + TN) * 100 /(TP + TN + FN + FP)
    print('FP: {}, FN: {}, TP: {}, TN: {}'.format(FP, FN, TP, TN))
    print('Acc: {:.3f}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(Acc, P, R, F1))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))

    #draw_evaluation("Evaluations", ['Acc', 'Precision', 'Recall', 'F1-measure'], [Acc, P, R, F1], 'evaluations', '%')