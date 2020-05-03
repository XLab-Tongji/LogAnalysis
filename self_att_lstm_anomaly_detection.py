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

# parameters for early prepare
log_file_dir = './Data/log/hdfs/'
# log file name used which is also used in ./Data/log/file_split
log_file_name = 'HDFS_split_40w'
log_fttree_out_directory = './Data/FTTreeResult-HDFS/clusters/'
# anomaly file name used which is also used in ./Data/log/file_split
anomaly_line_file = './Data/log/hdfs/HDFs_split_anomaly_40w'
wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec(0.1M)'
sequential_directory = './Data/FTTreeResult-HDFS/sequential_files/'
train_file_name = 'train_file'
test_file_name = 'test_file'
label_file_name = 'label_file'
pattern_vec_out_path = './Data/FTTreeResult-HDFS/pattern_vec'
split_degree = 0.9
# log file line used  which is also used in ./Data/log/file_split
log_line_num = 400000

# bi lstm only model parameters
window_length = 20
input_size = 300
hidden_size = 128
num_of_layers = 2
num_of_classes = 26
num_epochs = 10
batch_size = 1000
# for self att lstm
train_root_path = './Data/FTTreeResult-HDFS/self_att_lstm_model_train/'
model_out_path = train_root_path + 'sa_lstm_model_out/'
data_file = sequential_directory + train_file_name
pattern_vec_file = pattern_vec_out_path

# predict parameters
num_of_candidates = 8
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

def extract_feature_spilt_abnormal():
    hdfs_ft_preprocessor.preprocessor_hdfs_ft_split_abnormal(log_fttree_out_directory, anomaly_line_file, wordvec_file_path, sequential_directory, train_file_name, test_file_name, label_file_name, pattern_vec_out_path, split_degree, log_line_num)


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
    self_att_lstm_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, window_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt', sequential_directory + label_file_name, sequential_directory + test_file_name, num_of_candidates, pattern_vec_file)

def generate_seq_label(file_path, window_length, pattern_vec_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1
    num_of_sessions = 0
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            num_of_sessions += 1
            line = tuple(
                map(lambda n: tuple(map(float, n.strip().split())), [x for x in line.strip().split(',') if len(x) > 0]))
            if len(line) < window_length:
                continue
            for i in range(len(line) - window_length):
                label_line = []
                for j in range(window_length):
                    label_line.append(vec_to_class_type[line[i+j]])
                label_line.append(vec_to_class_type[line[i + window_length]])
                input_data.append(label_line)
    return input_data


def get_label_sequentials(sequential_out_file):
    vec_to_class_type = {}
    with open(pattern_vec_file, 'r') as pattern_file:
        i = 0
        for line in pattern_file.readlines():
            pattern, vec = line.split('[:]')
            pattern_vector = tuple(map(float, vec.strip().split(' ')))
            vec_to_class_type[pattern_vector] = i
            i = i + 1
    with open(sequential_out_file, 'w+') as file:
        sequence_data_set = generate_seq_label(data_file, window_length, pattern_vec_file)
        for line in sequence_data_set:
            for label in line:
                file.write(str(label))
                file.write(',')
            file.write('\n')


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