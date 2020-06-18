import os
import sys
sys.path.append('./')
from logparsing.drain.HDFS_drain import get_hdfs_drain_clusters
from extractfeature.hdfs_deeplog_preprocessor import hdfs_preprocessor
from anomalydetection.deeplog.Model1 import log_key_LSTM_train
from anomalydetection.deeplog.Model2 import variable_LSTM_train
from anomalydetection.deeplog import  log_predict


# log_train,log_test,logkey,logvalue
log = './Data/log/hdfs/HDFS_40w'
drain_out = './Data/Drain_HDFS/clusters/'
bin_dir = './HDFS_drain3_state.bin'
log_preprocessor_dir = './Data/Drain_HDFS/log_preprocessor'
model_dir = './Data/Drain_HDFS/deeplog_model_train/'

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

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def drain():
    get_hdfs_drain_clusters(log,drain_out,bin_dir)

def generate_logkey_and_value():
    hdfs_preprocessor()

# 训练
def train_model():
    train_model1()
    if use_model2:
        train_model2()

def train_model1():
    log_key_LSTM_train.train_model1(model_dir,log_preprocessor_dir,drain_out,model1_num_epochs,model1_batch_size,window_length,input_size,hidden_size,num_of_layers)

def train_model2():
    variable_LSTM_train.train_model2(model_dir,log_preprocessor_dir,model2_num_epochs,model2_batch_size,window_length,num_of_layers,learning_rate,hidden_size)

# 测试
def test_model():
    model1_name = 'Adam_batch_size=' + str(model1_batch_size) + ';epoch=' + str(model1_num_epochs) + '.pt'
    log_predict.do_predict(log_preprocessor_dir,drain_out,model_dir,model1_name,model2_num_epochs,window_length, input_size, hidden_size, num_of_layers, num_candidates, mse_threshold, use_model2)

#drain()
generate_logkey_and_value()
# train_model()
#test_model()