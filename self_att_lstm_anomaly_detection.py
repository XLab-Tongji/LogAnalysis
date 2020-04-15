# -*- coding: UTF-8 -*-
from extractfeature.k8s import log_preprocessor
from extractfeature.k8s import value_extract
import os
from logparsing.fttree import fttree
from extractfeature import hdfs_ft_preprocessor
from anomalydetection.self_att_lstm import self_att_lstm_train
from anomalydetection.self_att_lstm import self_att_lstm_predict

# parameters for early prepare
log_file_dir = './Data/log/hdfs/'
log_file_name = 'HDFS_split'
log_fttree_out_directory = './Data/FTTreeResult-HDFS/clusters/'
# anomaly file name used which is also used in ./Data/log/file_split
anomaly_line_file = './Data/log/hdfs/HDFs_split_anomaly'
wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec(0.1M)'
sequential_directory = './Data/FTTreeResult-HDFS/sequential_files/'
train_file_name = 'train_file'
test_file_name = 'test_file'
label_file_name = 'label_file'
pattern_vec_out_path = './Data/FTTreeResult-HDFS/pattern_vec'
split_degree = 0.2
# log file line used  which is also used in ./Data/log/file_split
log_line_num = 200000

# bi lstm only model parameters
window_length = 5
input_size = 300
hidden_size = 40
num_of_layers = 3
num_of_classes = 61
num_epochs = 100
batch_size = 200
# for self att lstm
train_root_path = './Data/FTTreeResult-HDFS/self_att_lstm_model_train/'
model_out_path = train_root_path + 'sa_lstm_model_out/'
data_file = sequential_directory + train_file_name
pattern_vec_file = pattern_vec_out_path

# predict parameters
num_of_candidates = 3
# log anomaly sequential model parameters

if not os.path.exists(log_fttree_out_directory):
    os.makedirs(log_fttree_out_directory)
if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


def pattern_extract():
    fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_directory, 5, 4, 2)

# 同时生成train file 和 test file好点
def extract_feature():
    hdfs_ft_preprocessor.preprocessor_hdfs_ft(log_fttree_out_directory, anomaly_line_file, wordvec_file_path, sequential_directory, train_file_name, test_file_name, label_file_name, pattern_vec_out_path, split_degree, log_line_num)


def pattern_extract_test():
    fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_directory, 5, 4, 2)


def extract_feature_test():
    hdfs_ft_preprocessor.preprocessor_hdfs_ft(log_fttree_out_directory, anomaly_line_file, wordvec_file_path, sequential_directory, 'train_file')


def train_model():
    #log_anomaly_sequential_train.train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, data_file, pattern_vec_file)
    self_att_lstm_train.train_model(window_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, data_file, pattern_vec_file)


def test_model():
    # do something
    #log_anomaly_sequential_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length, model_out_path + 'Adam_batch_size=200;epoch=200.pt', sequential_directory + label_file_name, sequential_directory + test_file_name, 3, pattern_vec_file)
    self_att_lstm_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length, model_out_path + 'Adam_batch_size=200;epoch=' + str(num_epochs) + '.pt', sequential_directory + label_file_name, sequential_directory + test_file_name, num_of_candidates, pattern_vec_file)


#extract_feature()
train_model()
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

# -*- coding: UTF-8 -*-