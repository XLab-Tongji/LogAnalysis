# -*- coding: UTF-8 -*-

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
from anomalydetection.att_all_you_need.encoder_self_att_train import Encoder
from anomalydetection.att_all_you_need.encoder_self_att_train import make_src_mask
from anomalydetection.att_all_you_need.encoder_self_att_train import get_batch_semantic_with_mask
# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# len(line) < window_length



def load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path, dropout, num_of_heads, pf_dim):

    model1 = Encoder(input_size, num_classes, hidden_size, num_layers, num_of_heads, pf_dim, dropout, device).to(device)
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


def do_predict(input_size, hidden_size, num_layers, num_classes, sequence_length, model_path, test_file_path, batch_size, pattern_vec_json, dropout, num_of_heads, pf_dim):

    sequential_model = load_sequential_model(input_size, hidden_size, num_layers, num_classes, model_path, dropout, num_of_heads, pf_dim)

    start_time = time.time()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # create data set
    sequence_data_set = generate_robust_seq_label(test_file_path, sequence_length)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=False, pin_memory=False)

    print('predict start')
    with torch.no_grad():
        for step, (seq, label) in enumerate(data_loader):
            # first traverse [0, window_size)
            batch_data, mask_data = get_batch_semantic_with_mask(seq, pattern_vec_json)
            seq = torch.tensor(batch_data)
            mask_data = torch.tensor(mask_data)
            seq = seq.view(-1, sequence_length, input_size).to(device)
            #label = torch.tensor(label).view(-1).to(device)
            output = sequential_model(seq, mask_data)[:, 0].clone().detach().cpu().numpy()
            predicted = (output > 0.5).astype(int)
            label = np.array([y for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
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