# -*- coding: UTF-8 -*-

import os
from logparsing.fttree import fttree
from extractfeature import hdfs_ft_preprocessor
from anomalydetection.loganomaly import log_anomaly_sequential_train
from anomalydetection.loganomaly import log_anomaly_sequential_predict
from anomalydetection.robust import bi_lstm_att_train
from anomalydetection.robust import bi_lstm_att_predict

# parameters for early prepare
'''log_file_dir = './Data/log/hdfs/'
log_file_name = 'HDFS_split'
log_fttree_out_directory = './Data/FTTreeResult-HDFS/clusters/'
# anomaly file name used which is also used in ./Data/log/file_split
anomaly_line_file = './Data/log/hdfs/HDFs_split_anomaly'
wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec(0.1M)'
sequential_directory = './Data/FTTreeResult-HDFS/sequential_files/'
train_file_name = 'train_file'
test_file_name = 'test_file'
label_file_name = 'label_file'
pattern_vec_out_path = './Data/FTTreeResult-HDFS/pattern_vec'''

temp_directory = './Data/logdeepdata/'
train_file_name = 'robust_log_train.csv'
test_file_name = 'robust_log_test.csv'
valid_file_name = 'robust_log_valid.csv'

# log anomaly sequential model parameters some parameter maybe changed to train similar models
sequence_length = 50
input_size = 300
hidden_size = 128
num_of_layers = 2
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 20
batch_size = 1000
# for robust attention bi
train_root_path = './Data/FTTreeResult-HDFS/robust_att_bi_model_train/'
model_out_path = train_root_path + 'model_out/'
train_file = temp_directory + train_file_name
pattern_vec_json = './Data/logdeepdata/event2semantic_vec.json'


# predict parameters
# log anomaly sequential model parameters

'''if not os.path.exists(log_fttree_out_directory):
    os.makedirs(log_fttree_out_directory)
if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)'''
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)


'''def pattern_extract():
    fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_directory, 5, 4, 2)

# 同时生成train file 和 test file好点
def extract_feature():
    hdfs_ft_preprocessor.preprocessor_hdfs_ft(log_fttree_out_directory, anomaly_line_file, wordvec_file_path, sequential_directory, train_file_name, test_file_name, label_file_name, pattern_vec_out_path, split_degree, log_line_num)


def pattern_extract_test():
    fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_directory, 5, 4, 2)


def extract_feature_test():
    hdfs_ft_preprocessor.preprocessor_hdfs_ft(log_fttree_out_directory, anomaly_line_file, wordvec_file_path, sequential_directory, 'train_file')
'''

def train_model():
    bi_lstm_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json)


def test_model():
    # do something
    bi_lstm_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt', temp_directory + test_file_name, batch_size, pattern_vec_json)

#pattern_extract()
#extract_feature()
#train_model()
train_model()
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

