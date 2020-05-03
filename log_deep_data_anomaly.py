# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
from extractfeature.k8s import log_preprocessor
from extractfeature.k8s import value_extract
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from logparsing.fttree import fttree
from extractfeature import hdfs_ft_preprocessor
from anomalydetection.self_att_lstm import self_att_lstm_train
from anomalydetection.self_att_lstm import self_att_lstm_predict

sequential_directory = './Data/logdeepdata/'
train_file_name = 'hdfs_train'
test_abnormal_name = 'hdfs_test_abnormal'
test_normal_name = 'hdfs_test_normal'
pattern_vec_out_path = './Data/FTTreeResult-HDFS/pattern_vec'


#  lstm att model parameters
window_length = 10
input_size = 1
hidden_size = 128
num_of_layers = 2
num_of_classes = 28
num_epochs = 20
batch_size = 2000
# for self att lstm
train_root_path = './Data/Logdeep_Result/self_att_lstm_model_train/'
model_out_path = train_root_path + 'sa_lstm_model_out/'
data_file = sequential_directory + train_file_name
pattern_vec_file = pattern_vec_out_path

# predict parameters
num_of_candidates = 8
# log anomaly sequential model parameters

if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)



def train_model():
    #log_anomaly_sequential_train.train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, data_file, pattern_vec_file)
    self_att_lstm_train.train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, data_file, pattern_vec_file)


def test_model():
    # do something
    #log_anomaly_sequential_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length, model_out_path + 'Adam_batch_size=200;epoch=200.pt', sequential_directory + label_file_name, sequential_directory + test_file_name, 3, pattern_vec_file)
    self_att_lstm_predict.do_log_deep_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt', sequential_directory + test_normal_name, sequential_directory + test_abnormal_name, num_of_candidates, pattern_vec_file)


#pattern_extract()
#extract_feature_spilt_abnormal()
#train_model()
#get_label_sequentials('./Data/FTTreeResult-HDFS/pattern_sequntials')
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

# -*- coding: UTF-8 -*-