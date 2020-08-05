# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd


import numpy as np

import random
import math
import time
import json
import os

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_src_mask(src, src_pad_idx):
    # src = [batch size, src len]

    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # src_mask = [batch size, 1, 1, src len]

    return src_mask.clone().detach().numpy().tolist()


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,  #
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=100):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Linear(input_dim, hid_dim)  #
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.output = nn.Linear(hid_dim, output_dim)  #

    def forward(self, src, src_mask):
        # src = [batch size, src len, input_dim] #
        # src_mask = [batch size, 1, 1, src len] # original comment is wrong but code is right


        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch size, src len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        output = self.output(src)  #
        output = torch.sigmoid(output[:, -1, :])  #
        return output


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]

        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


def generate_robust_seq_label(file_path, sequence_length, pattern_vec_file):
    num_of_sessions = 0
    input_data, output_data, mask_data = [], [], []
    train_file = pd.read_csv(file_path)
    for i in range(len(train_file)):
        num_of_sessions += 1
        line = [int(id) for id in train_file["Sequence"][i].split(' ')]
        line = line[0:sequence_length]
        if len(line) < sequence_length:
            line.extend(list([0]) * (sequence_length - len(line)))
        input_data.append(line)
        output_data.append(int(train_file["label"][i]))
    data_set = TensorDataset(torch.tensor(input_data), torch.tensor(output_data))
    return data_set


def get_batch_semantic_with_mask(seq, pattern_vec_file):
    with open(pattern_vec_file, 'r') as pattern_file:
        class_type_to_vec = json.load(pattern_file)
    batch_data = []
    mask_data = []
    for s in seq:
        semantic_line = []
        for event in s.numpy().tolist():
            if event == 0:
                semantic_line.append([-1] * 300)
            else:
                semantic_line.append(class_type_to_vec[str(event)])
        batch_data.append(semantic_line)
        mask = make_src_mask(s, 0)
        mask_data.append(mask)
    return batch_data, mask_data


def train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, root_path, model_output_directory, data_file, pattern_vec_file, dropout, num_of_heads, pf_dim):
    print("Train num_classes: ", num_of_classes)
    model = Encoder(input_size, num_of_classes, hidden_size, num_of_layers, num_of_heads, pf_dim, dropout, device).to(device)
    # create data set
    sequence_data_set = generate_robust_seq_label(data_file, sequence_length, pattern_vec_file)
    # create data_loader
    data_loader = DataLoader(dataset=sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=False)

    # Loss and optimizer  classify job
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            batch_data, mask_data = get_batch_semantic_with_mask(seq, pattern_vec_file)
            seq = torch.tensor(batch_data)
            #print(seq.shape)
            seq = seq.clone().detach().view(-1, sequence_length, input_size).to(device)
            #print(seq.shape)
            output = model(seq, torch.tensor(mask_data))

            loss = criterion(output.squeeze(-1), label.float().to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        if (epoch + 1) % num_epochs == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    print('Training finished')