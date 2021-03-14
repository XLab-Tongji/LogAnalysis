# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-

import os
import torch
import numpy as np
import random

from anomalydetection.att_all_you_need import encoder_self_att_train
from anomalydetection.att_all_you_need import encoder_self_att_predict
from extractfeature import hdfs_selfatt_preprocessor

# parameters for early prepare
logparser_structed_file = './Data/logparser_result/Drain/HDFS.log_structured.csv'
logparser_event_file = './Data/logparser_result/Drain/HDFS.log_templates.csv'
anomaly_label_file = './Data/log/hdfs/anomaly_label.csv'
train_root_path = './Data/DrainResult-HDFS/att_all_you_need/'

sequential_directory = train_root_path + 'sequential_files/'
train_file_name = 'encoder_train_file'
test_file_name = 'encoder_test_file'
valid_file_name = 'encoder_valid_file'

wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec'
pattern_vec_out_path = train_root_path + 'pattern_vec(l-ti-advance)'
#pattern_vec_out_path = train_root_path + 'pattern_vec(l-ti)'
variable_symbol = '<*>'

# my encoder
sequence_length = 50
input_size = 300
hidden_size = 256
num_of_layers = 4
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 100
batch_size = 1024


model_out_path = train_root_path + 'model_out/'
train_file = sequential_directory + train_file_name
pattern_vec_json = pattern_vec_out_path
dropout = 0.2
num_of_heads = 8
pf_dim = 512


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

# predict parameters
# log anomaly sequential model parameters

if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def extract_feature():
    hdfs_selfatt_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file, anomaly_label_file, sequential_directory, train_file_name, valid_file_name, test_file_name, wordvec_file_path, pattern_vec_out_path, variable_symbol)


def train_model():
    encoder_self_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json, dropout, num_of_heads, pf_dim)

def pattern_to_vec():
    hdfs_selfatt_preprocessor.pattern_to_vec_tf_idf_from_log(logparser_event_file, wordvec_file_path, pattern_vec_out_path, variable_symbol)

def test_model():
    # do something
    encoder_self_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + ';sequence=' + str(sequence_length) + '.pt', sequential_directory + test_file_name, batch_size, pattern_vec_json, dropout, num_of_heads, pf_dim)


print(pattern_vec_out_path)

set_seed(1) #4 （98.094% 12.3修改之前的版本 也就是0.1m直接读的版本 没有用徐博的代码的那个版本）
            #1  98.639% 修改后版本 全
extract_feature()
pattern_to_vec()
train_model()
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

