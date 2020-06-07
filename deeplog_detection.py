import os
from logparsing.fttree import fttree
from extractfeature import hdfs_fs_deeplog_preprocessor
from anomalydetection.deeplog.Model1 import log_key_LSTM_train
from anomalydetection.deeplog.Model2 import variable_LSTM_train
from anomalydetection.deeplog import  log_predict

# 原始日志文件
log_file_dir = './Data/log/hdfs/'
log_file_name = 'HDFS_split'
log_file_abnormal_label = 'HDFS_split_anomaly'
# FT-tree
log_result = './Data/FTTreeResult-HDFS/'
log_fttree_out_dir = log_result+'clusters/'
# log_train,log_test,logkey,logvalue
log_preprocessor_dir = log_result+'deeplog_files/'
# model
model_dir = log_result+'deeplog_model_train/'
# train parameters
window_length = 4
input_size = 1
hidden_size = 20
num_of_layers = 3
model1_num_epochs = 100
model1_batch_size = 200
model2_num_epochs = 50
model2_batch_size = 20
learning_rate = 0.01
num_candidates = 3
mse_threshold = 0.1
# 是否使用模型二
use_model2 = False

if not os.path.exists(log_result):
    os.makedirs(log_result)
if not os.path.exists(log_fttree_out_dir):
    os.makedirs(log_fttree_out_dir)
if not os.path.exists(log_preprocessor_dir):
    os.makedirs(log_preprocessor_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# FT-tree
def pattern_extract():
    fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_dir, 5, 4, 2)

# 将原日志文件分成训练集和测试集两部分
def log_split():
    hdfs_fs_deeplog_preprocessor.log_split(log_file_dir,log_file_name,log_file_abnormal_label,log_preprocessor_dir)

# 生成log_key
def generate_log_key():
    hdfs_fs_deeplog_preprocessor.generate_log_key(log_file_dir,log_file_abnormal_label,log_preprocessor_dir,log_fttree_out_dir)

# 提取并处理log_value
def generate_log_value():
    hdfs_fs_deeplog_preprocessor.generate_log_value(log_file_dir,log_file_name,log_file_abnormal_label,log_preprocessor_dir,log_fttree_out_dir)

# 训练
def train_model():
    train_model1()
    if use_model2:
        train_model2()

def train_model1():
    log_key_LSTM_train.train_model1(model_dir,log_preprocessor_dir,log_fttree_out_dir,model1_num_epochs,model1_batch_size,window_length,input_size,hidden_size,num_of_layers)

def train_model2():
    variable_LSTM_train.train_model2(model_dir,log_preprocessor_dir,model2_num_epochs,model2_batch_size,window_length,num_of_layers,learning_rate,hidden_size)

# 测试
def test_model():
    model1_name = 'Adam_batch_size=' + str(model1_batch_size) + ';epoch=' + str(model1_num_epochs) + '.pt'
    log_predict.do_predict(log_preprocessor_dir,log_fttree_out_dir,model_dir,model1_name,model2_num_epochs,window_length, input_size, hidden_size, num_of_layers, num_candidates, mse_threshold, use_model2)


#pattern_extract()
#log_split()
#generate_log_key()
#generate_log_value()
train_model()
test_model()