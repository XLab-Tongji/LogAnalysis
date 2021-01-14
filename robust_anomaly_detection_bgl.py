# -*- coding: UTF-8 -*-

import os
from logparsing.fttree import fttree
from extractfeature import bgl_selfatt_robust_preprocessor
from anomalydetection.loganomaly import log_anomaly_sequential_train
from anomalydetection.loganomaly import log_anomaly_sequential_predict
from anomalydetection.robust import bi_lstm_att_train
from anomalydetection.robust import bi_lstm_att_predict
from logparsing.converter import eventid2number
import numpy as np
import random
import torch


# parameters for early prepare
window_length = 200
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
pattern_vec_out_path = './Data/DrainResult-BGL/robust/pattern_vec'
variable_symbol = '<*> '



# log anomaly sequential model parameters some parameter maybe changed to train similar models
sequence_length = window_length
input_size = 300
hidden_size = 32
num_of_layers = 2
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 12
batch_size = 64
# for robust attention bi
train_root_path = './Data/DrainResult-BGL/robust/'
model_out_path = train_root_path + 'model_out/'
train_file = sequential_directory + train_file_name
pattern_vec_json = pattern_vec_out_path


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


def extract_feature():
    bgl_selfatt_robust_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file, sequential_directory, train_file_name, valid_file_name, test_file_name, wordvec_file_path, pattern_vec_out_path, variable_symbol, window_length, step_length)


def train_model():
    bi_lstm_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json)


def pattern_to_vec():
    bgl_selfatt_robust_preprocessor.pattern_to_vec_robust(logparser_event_file, wordvec_file_path, pattern_vec_out_path, variable_symbol)


def test_model():
    # do something
    bi_lstm_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + ';sequence=' + str(sequence_length) + '.pt', sequential_directory + test_file_name, batch_size, pattern_vec_json)

set_seed(2)
#eventid2number.add_numberid(logparser_event_file)
#extract_feature()
#pattern_to_vec()
train_model()
test_model()



