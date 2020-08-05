# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
import torch
import json
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import time
import random
from torch.utils.data import TensorDataset, DataLoader
from anomalydetection.robust.bi_lstm_att_train import Model

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

    model1 = Model(input_size, hidden_size, num_layers, num_classes, if_bidirectional=True, batch_size=0).to(device)
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


def generate_robust_seq_label(file_path, sequence_length):
    num_of_sessions = 0
    input_data, output_data, mask_data = [], [], []
    train_file = pd.read_csv(file_path)
    i = 0
    while i < len(train_file):
        num_of_sessions += 1
        line = [int(id) for id in train_file["Sequence"][i].strip().split(' ')]
        line = line[0:sequence_length]
        if len(line) < sequence_length:
            line.extend(list([0]) * (sequence_length - len(line)))
        input_data.append(line)
        output_data.append(int(train_file["label"][i]))
        i += 1
    data_set = TensorDataset(torch.tensor(input_data), torch.tensor(output_data))
    return data_set


def get_batch_semantic(seq, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        class_type_to_vec = json.load(pattern_file)
    batch_data = []
    for s in seq:
        semantic_line = []
        for event in s.numpy().tolist():
            if event == 0:
                semantic_line.append([-1] * 300)
            else:
                semantic_line.append(class_type_to_vec[str(event)])
        batch_data.append(semantic_line)
    return batch_data


def do_predict(input_size, hidden_size, num_layers, num_classes, sequence_length, model_path, test_file_path, batch_size, pattern_vec_json):

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # create data set
    sequence_data_set = generate_robust_seq_label(test_file_path, sequence_length)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    print('predict start')
    with torch.no_grad():
        count = 0
        for step, (seq, label) in enumerate(data_loader):
            batch_data = get_batch_semantic(seq, pattern_vec_json)
            seq = torch.tensor(batch_data)
            seq = seq.view(-1, sequence_length, input_size).to(device)
            output = sequential_model(seq)[:, 0].cpu().clone().detach().numpy()
            predicted = (output > 0.2).astype(int)
            label = np.array([y for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
            count += 1
            if count > 100000:
                break
    ALL = TP + TN + FP + FN
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