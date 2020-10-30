# -*- coding: UTF-8 -*-

import os
from logparsing.fttree import fttree
from extractfeature import bgl_selfatt_robust_preprocessor
from anomalydetection.loganomaly import log_anomaly_sequential_train
from anomalydetection.loganomaly import log_anomaly_sequential_predict
from anomalydetection.att_all_you_need import encoder_self_att_train
from anomalydetection.att_all_you_need import encoder_self_att_predict
from logparsing.converter import eventid2number
import numpy as np
import random
import torch


# parameters for early prepare
window_length = 150
step_length = 6
logparser_structed_file = './Data/logparser_result/Drain/BGL.log_structured.csv'
logparser_event_file = './Data/logparser_result/Drain/BGL.log_templates.csv'

sequential_directory = './Data/DrainResult-BGL/att_all_you_need/sequential_files/'
train_file_name = 'encoder_train_file_bgl' + '_window'+str(window_length)+'_step'+str(step_length)
test_file_name = 'encoder_test_file_bgl' + '_window' + str(window_length)+'_step'+str(step_length)
valid_file_name = 'encoder_valid_file_bgl' + '_window' + str(window_length)+'_step'+str(step_length)
'''
sequential_directory = './Data/DrainResult-BGL/robust/sequential_files/'
train_file_name = 'robust_train_file_bgl' + '_window'+str(window_length)+'_step'+str(step_length)
test_file_name = 'robust_test_file_bgl' + '_window' + str(window_length)+'_step'+str(step_length)
valid_file_name = 'robust_valid_file_bgl' + '_window' + str(window_length)+'_step'+str(step_length)
'''
wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec'
pattern_vec_out_path = './Data/DrainResult-BGL/att_all_you_need/pattern_vec'
train_root_path = './Data/DrainResult-BGL/att_all_you_need/'
variable_symbol = '<*> '



# log anomaly sequential model parameters some parameter maybe changed to train similar models
# my encoder
sequence_length = window_length
input_size = 300
hidden_size = 256
num_of_layers = 2
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 10
batch_size = 512


model_out_path = train_root_path + 'model_out/'
train_file = sequential_directory + train_file_name
pattern_vec_json = pattern_vec_out_path
dropout = 0.2
num_of_heads = 8
pf_dim = 512


# predict parameters

# log anomaly sequential model parameters

'''if not os.path.exists(log_fttree_out_directory):
    os.makedirs(log_fttree_out_directory)'''
if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def extract_feature():
    bgl_selfatt_robust_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file, sequential_directory, train_file_name, valid_file_name, test_file_name, wordvec_file_path, pattern_vec_out_path, variable_symbol, window_length, step_length)


def pattern_to_vec():
    bgl_selfatt_robust_preprocessor.pattern_to_vec_tf_idf_from_log(logparser_event_file, wordvec_file_path, pattern_vec_out_path, variable_symbol)


def train_model():
    encoder_self_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json, dropout, num_of_heads, pf_dim)


def test_model():
    # do something
    encoder_self_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + ';sequence=' + str(sequence_length) + '.pt', sequential_directory + test_file_name, batch_size, pattern_vec_json, dropout, num_of_heads, pf_dim)


print(pattern_vec_out_path)

set_seed(2)

#extract_feature()
#pattern_to_vec()
train_model()
test_model()