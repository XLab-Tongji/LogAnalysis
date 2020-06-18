import os
import linecache
import sys
import time
import torch
import torch.nn as nn
path = "C:\\study\\code\\LogAnalysis\\"
sys.path.append(path)
from logparsing.drain.HDFS_drain import get_hdfs_drain_clusters
from anomalydetection.deeplog.Model1.log_key_LSTM_train import Model as Model1
from anomalydetection.deeplog.Model2.variable_LSTM_train import Model as Model2
import sys
import shutil

log_detect_name  = sys.argv[1]
use_model2 = sys.argv[2]

# log_detect_name = 'detect.log'
# use_model2 = '1'

log_file_dir = path+'/Data/log/hdfs/'
log_file_name = 'HDFS_split'
base_dir = path+'/java/'
# log_detect
log_detect_dir = base_dir+'/detect_log/'
# Drian
drain_out = log_detect_dir + 'clusters/'
bin_dir = path + 'HDFS_drain3_state.bin'

WORD_VECTOR_FILE = path + '/Data/log/hdfs/word2vec_HDFS_40w'
# model
model_dir =  path+'Data/Drain_HDFS/'+'deeplog_model_train/'
N_Clusters = 31
window_length = 4
input_size = 1
hidden_size = 20
num_layers = 3
model1_num_epochs = 100
model1_batch_size = 200
model2_num_epochs = 50
model2_batch_size = 20
learning_rate = 0.01
num_candidates = 3
mse_threshold = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shutil.rmtree(drain_out)
os.makedirs(drain_out)
shutil.rmtree(log_detect_dir+'logvalue/')
os.makedirs(log_detect_dir+'logvalue/')


def load_word_vector():
    word_to_vector = {}
    with open(WORD_VECTOR_FILE, 'r') as r:
        for line in r.readlines():
            list_line = line.split(' ')
            value = list(map(float, list_line[1:]))
            key = list_line[0]
            word_to_vector[key] = value
    return word_to_vector

def get_sentence_vector(word_to_vector, sentence):
    words = sentence.split(' ')
    old_vector = [0.0 for i in range(10)]
    for word in words:
        if word not in word_to_vector.keys():
            another_vector = [0.0 for i in range(10)]
        else:
            another_vector = word_to_vector[word]
        new_vector = []
        for i, j in zip(old_vector, another_vector):
            new_vector.append(i + j)
        old_vector = new_vector

    word_count = len(words)
    for idx, value in enumerate(old_vector):
        old_vector[idx] = value / word_count
    vector_str = list(map(str, old_vector))
    sentence_vector = ','.join(vector_str)
    return sentence_vector

def generate(name,window_length):
    log_keys_sequences=list()
    length=0
    with open(name, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_length + 1 - len(line))
            # for i in range(len(line) - window_size):
            #     inputs.add(tuple(line[i:i+window_size]))
            # log_keys_sequences[tuple(line)] = log_keys_sequences.get(tuple(line), 0) + 1
            log_keys_sequences.append(tuple(line))
            length+=1
    return log_keys_sequences,length

def load_model1(model_dir,model_name,input_size, hidden_size, num_layers):
    value_length_of_key = [10] * (31 + 1)
    num_classes = len(value_length_of_key)
    print("Model1 num_classes: ", num_classes)
    model1_dir = model_dir + 'model1/'
    model_path = model1_dir + model_name
    model1 = Model1(input_size, hidden_size, num_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model_path, map_location='cpu'))
    model1.eval()
    print('model_path: {}'.format(model_path))
    return model1


def load_model2(model_dir,epoch,input_size, hidden_size, num_layers):
    model2_dir = model_dir+ 'model2/'
    model2 = []
    value_length_of_key = [10] * (31 + 1)
    for i in range(len(value_length_of_key)):
        if value_length_of_key[i] == 0:
            model2.append(None)
            continue
        input_size = value_length_of_key[i]
        out_size = input_size
        model_name = str(i+1) + '_epoch=' + str(epoch)+ '.pt'
        model_path = model2_dir + str(i+1) + '/' + model_name
        if not os.path.exists(model_path):
            model2.append(None)
            continue
        model = Model2(input_size, hidden_size, num_layers, out_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print('model_path: {}'.format(model_path))
        model2.append(model)
    return model2

def generate_log_key_and_value():
    print("generating log key...")
    get_hdfs_drain_clusters(log_detect_dir+log_detect_name, drain_out,bin_dir)
    log_to_key = {}
    for i in range(0,N_Clusters):
        if os.path.exists(drain_out+str(i+1)):
            with open(drain_out+str(i+1),'r') as file:
                for line in (file.readline().split()):
                    log_to_key[line] = i+1
    print(log_to_key)
    # with open(log_detect_dir+'logkey.txt','w') as file:
    #     for i in range(0,len(log_to_key)):
    #         file.write(str(log_to_key[str(i)]))
    #         file.write(" ")

    print("generating log value...")
    word_to_vector = load_word_vector()
    logkey_to_logvalues = [[] for i in range(N_Clusters + 1)]
    logkeys = []
    with open(log_detect_dir+log_detect_name,'r') as file:
        lines = file.readlines()
        for i in range(0,len(lines)):
            logkey = log_to_key[str(i)]
            logkeys.append(logkey)
            vector = get_sentence_vector(word_to_vector,lines[i])
            logkey_to_logvalues[logkey].append(vector)
    logkey_line = ' '.join(str(logkey) for logkey in logkeys)
    logkey_writelist = []
    logkey_to_logvalue_writelist = [[] for i in range(N_Clusters + 1)]
    logkey_writelist.append(logkey_line + '\n')
    for logkey in range(1, N_Clusters + 1):
        if len(logkey_to_logvalues[logkey]) == 0:
            logvalue_line = '-1'
        else:
            logvalue_line = ' '.join(logkey_to_logvalues[logkey])
        logkey_to_logvalue_writelist[logkey].append(logvalue_line + '\n')
    print(logkey_writelist)
    with open(log_detect_dir+'logkey.txt', 'w') as f:
        f.writelines(logkey_writelist)
    os.makedirs(log_detect_dir+'logvalue/',exist_ok=True)
    for logkey in range(1, N_Clusters + 1):
        LOGVALUE_FILE = str(logkey)
        with open(log_detect_dir+'logvalue/' + LOGVALUE_FILE, 'w') as f:
            f.writelines(logkey_to_logvalue_writelist[logkey])

def log_predict(use_model2):
    model1_name = 'Adam_batch_size=' + str(model1_batch_size) + ';epoch=' + str(model1_num_epochs) + '.pt'
    model1 = load_model1(model_dir, model1_name, input_size, hidden_size, num_layers)
    model2 = load_model2(model_dir, model2_num_epochs, 10, hidden_size, num_layers)
    start_time = time.time()
    criterion = nn.MSELoss()
    test_normal_loader, test_normal_length = generate(log_detect_dir + 'logkey.txt', window_length)
    print('predict start')
    FP=0
    with torch.no_grad():
        for line_num, line in enumerate(test_normal_loader):
            model1_success = False
            for i in range(len(line) - window_length - 1):
                seq0 = line[i:i + window_length]
                label = line[i + window_length]
                seq0 = torch.tensor(seq0, dtype=torch.float).view(
                    -1, window_length, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model1(seq0)
                predicted = torch.argsort(output,1)[0][-num_candidates:]
                if label not in predicted:
                    FP+=1
                    print(FP)
                    model1_success = True
                    break
            if (model1_success):
                continue

            if use_model2=='1':
                seq = []
                for i in range(31):
                    with open(log_detect_dir + '/logvalue/' + str(i + 1), 'r')as f:
                        key_values = f.readlines()
                        key_values = key_values[line_num].strip('\n')
                        if (key_values == '-1'):
                            continue
                        seq.append(key_values.split(' '))
                # 将字符串转为数字
                for k1 in range(len(seq)):
                    for k2 in range(len(seq[k1])):
                        seq[k1][k2] = seq[k1][k2].strip('\n')
                        seq[k1][k2] = seq[k1][k2].split(',')
                        for k3 in range(len(seq[k1][k2])):
                            if (seq[k1][k2][k3] != ''):
                                seq[k1][k2][k3] = float(seq[k1][k2][k3])

                # 补全
                for i in range(len(seq)):
                    if (len(seq[i]) < window_length + 1):
                        for j in range(window_length + 1 - len(seq[i])):
                            seq[i].append([0.0] * 10)
                model2_success = False
                # 预测
                for i in range(len(seq)):
                    if (model2[i] == None):
                        continue
                    for j in range(len(seq[i]) - window_length):
                        seq2 = seq[i][j:j + window_length]
                        label2 = seq[i][j + window_length]

                        seq2 = torch.tensor(seq2, dtype=torch.float).view(
                            -1, window_length, 10).to(device)
                        label2 = torch.tensor(label, dtype=torch.float).view(-1).to(device)
                        output = model2[i](seq2)
                        mse = criterion(output[0], label2.to(device))
                        if mse > mse_threshold:
                            FP += 1
                            model2_success = True
                            break
                    if (model2_success):
                        break
    if(FP==1):
        print("predict result: abnormal")
    else:
        print("predict result: normal")





generate_log_key_and_value()
log_predict(use_model2)








