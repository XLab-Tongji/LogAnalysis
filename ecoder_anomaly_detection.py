# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-

import os
from logparsing.fttree import fttree
from extractfeature import hdfs_ft_preprocessor
from anomalydetection.loganomaly import log_anomaly_sequential_train
from anomalydetection.loganomaly import log_anomaly_sequential_predict
from anomalydetection.att_all_you_need import encoder_self_att_train
from anomalydetection.att_all_you_need import encoder_self_att_predict

# parameters for early prepare

temp_directory = './Data/logdeepdata/'
train_file_name = 'robust_log_train.csv'
test_file_name = 'robust_log_test.csv'
valid_file_name = 'robust_log_valid.csv'

# log anomaly sequential model parameters some parameter maybe changed to train similar models
sequence_length = 50
input_size = 300
hidden_size = 256
num_of_layers = 4
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 20
batch_size = 1000
# for robust attention bi
train_root_path = './Data/FTTreeResult-HDFS/att_all_you_need/'
model_out_path = train_root_path + 'model_out/'
train_file = temp_directory + train_file_name
pattern_vec_json = './Data/logdeepdata/event2semantic_vec.json'
dropout = 0.5
num_of_heads = 8
pf_dim = 512


# predict parameters
# log anomaly sequential model parameters

if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def train_model():
    encoder_self_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json, dropout, num_of_heads, pf_dim)


def test_model():
    # do something
    encoder_self_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt', temp_directory + test_file_name, batch_size, pattern_vec_json, dropout, num_of_heads, pf_dim)

#pattern_extract()
#extract_feature()
#train_model()
#train_model()
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

