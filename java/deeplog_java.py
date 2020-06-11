import os
import linecache
import sys
import time
import torch
import torch.nn as nn
path = "C:\\study\\code\\LogAnalysis\\"
sys.path.append(path)
from anomalydetection.deeplog.Model1.log_key_LSTM_train import Model as Model1
from anomalydetection.deeplog.Model2.variable_LSTM_train import Model as Model2
import sys

log_detect_name  = sys.argv[1]
use_model2 = sys.argv[2]

log_file_dir = path+'/Data/log/hdfs/'
log_file_name = 'HDFS_split'
# log_train
base_dir = path+'/java/'
# log_detect
log_detect_dir = base_dir+'/detect_log/'
# FT-tree
train_fttree_out_dir = path+'Data/FTTreeResult-HDFS/clusters/'
# log_train,log_test,logkey,logvalue
WORD2VEC_FILE =base_dir+'word2vec'
log_key_detect = log_detect_dir + 'log_key_detect.txt'
log_value_detect = log_detect_dir +'log_value_detect/'
# model
model_dir =  path+'Data/FTTreeResult-HDFS/'+'deeplog_model_train/'


def generate_log_key_and_value():
    print("generating log key...")
    log_detect = []
    with open(log_detect_dir+log_detect_name,'r') as file:
        for line in file.readlines():
            log_detect.append(line.split('\n')[0])
    log = []
    with open(log_file_dir+log_file_name,'r') as file:
        for line in file.readlines():
            log.append(line.split('\n')[0])
    train_key_pattern = []
    with open(train_fttree_out_dir+'key_pattern.txt','r') as file:
        for line in file.readlines():
            train_key_pattern.append(line.split('\n')[0])
    #print(train_key_pattern)
    N_CLUSTER = len(train_key_pattern)
    log_index = log.index(log_detect[0])
    #print(log_index)
    all_log_key = {}
    for i in range(0,N_CLUSTER):
        with open(train_fttree_out_dir + str(i+1), 'r') as file:
            keys = file.readlines()[1].split()
            for key in keys:
                all_log_key[int(key)]=i+1
    log_key = {}
    for i in range(0,len(log_detect)):
        log_key[i+1] = all_log_key[i+log_index]
    #print(log_key)
    with open(log_key_detect,'w') as file:
        for i in range(0,len(log_detect)):
            file.write(str(log_key[i+1])+' ')

    print("generating log value...")
    log_list = []
    word_vector = {}
    word2vec = WORD2VEC_FILE
    if not os.path.exists(log_value_detect):
        os.makedirs(log_value_detect)
    with open(word2vec, 'r') as r:
        for line in r.readlines():
            list_line = line.split(' ')
            value = list(map(float, list_line[1:]))
            key = list_line[0]
            word_vector[key] = value
    for i in range(0, len(log_detect)):
        template = train_key_pattern[log_key[i+1]-1]
        mytemplate = template.split()
        mytemplate.append('INFO')
        mylog = log[i].split()
        log_str = ""
        for word in mylog:
            if word not in mytemplate:
                log_str += word + " "
        log_list.append(log_str)
    new_clusters = []
    for i in range(N_CLUSTER):
        new_clusters.append([])
    for i in range(0,len(log_key)):
        new_clusters[log_key[i+1]-1].append(i)
    #print(new_clusters)

    for i in range(N_CLUSTER):
        #print("process:", i)
        out_path = log_value_detect+ str(i + 1) + ".txt"
        write_list = []
        for t in new_clusters[i]:
            s = int(t)
            output = calc_sentence_vector(log_list[s], word_vector)
            write_list.append(output)
        with open(out_path, mode='w', encoding='utf-8') as f:
            f.write('\n'.join(write_list))

def calc_sentence_vector(sentence, word_vector):
    VECTOR_DIMENSION = 10
    words = sentence.split(' ')
    old_vector = [0.0 for i in range(VECTOR_DIMENSION)]
    for word in words:
        if word not in word_vector.keys():
            another_vector = [0.0 for i in range(VECTOR_DIMENSION)]
        else:
            another_vector = word_vector[word]
        new_vector = []
        for i, j in zip(old_vector, another_vector):
            new_vector.append(i + j)
        old_vector = new_vector
    word_count = len(words)
    for idx, value in enumerate(old_vector):
        old_vector[idx] = value / word_count
    vector_str = list(map(str, old_vector))
    output = ','.join(vector_str)
    return output

def log_predict(use_model2):
    window_length = 4
    input_size = 1
    hidden_size = 20
    num_of_layers = 3
    model1_num_epochs = 100
    model1_batch_size = 200
    model2_num_epochs = 5
    model2_batch_size = 20
    learning_rate = 0.01
    num_candidates = 3
    mse_threshold = 0.1
    model1_name = 'Adam_batch_size=' + str(model1_batch_size) + ';epoch=' + str(model1_num_epochs) + '.pt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    criterion = nn.MSELoss()
    train_key_pattern = []
    with open(train_fttree_out_dir+'key_pattern.txt','r') as file:
        for line in file.readlines():
            train_key_pattern.append(line.split('\n')[0])
    num_classes = len(train_key_pattern)+2
    model1_dir = model_dir + 'model1/'
    model1_path = model1_dir + model1_name
    model1 = Model1(input_size, hidden_size, num_of_layers, num_classes).to(device)
    model1.load_state_dict(torch.load(model1_path, map_location='cpu'))
    model1.eval()
    model2_dir = model_dir+ 'model2/'
    model2 = []
    for i in range(num_classes):
        input_size = 1
        out_size = input_size
        model_name = str(i+1) + '_epoch=' + str(model2_num_epochs)+ '.pt'
        model_path = model2_dir + str(i+1) + '/' + model_name
        if not os.path.exists(model_path):
            model2.append(None)
            continue
        model = Model2(input_size, hidden_size, num_of_layers, out_size).to(device)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print('model_path: {}'.format(model_path))
        model2.append(model)

    log_keys_sequences = list()
    with open(log_key_detect, 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_length + 1 - len(line))
            log_keys_sequences.append(tuple(line))

    pattern2value = []
    file_names = os.listdir(log_value_detect)
    pattern2value.append([])
    for i in range(len(file_names)):
        pattern2value.append([])
        with open(log_value_detect + str(i+1) + ".txt", 'r') as in_text:
            for line in in_text.readlines():
                line = list(map(lambda n: n, map(float, line.split(','))))
                pattern2value[i+1].append(line)
    #print(pattern2value)
    pattern_index = [0] * len(pattern2value)

    print('predict start')
    abnormal = []
    with torch.no_grad():
        count_num = 0
        count = 0
        for line in log_keys_sequences:
            i = 0
            for ii in range(window_length):
                if ii < len(line):
                    pattern_index[line[ii]] += 1
            print(pattern_index)
            while i < len(line) - window_length:
                lineNum = i + window_length + 1
                seq = line[i:i + window_length]
                label = line[i + window_length]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_length, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model1(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                now_pattern_index = pattern_index[label]
                # print('{} - predict result: {}, true label: {}'.format(lineNum, predicted, label))
                if label not in predicted: #异常
                    count+=1
                    abnormal.append(lineNum)
                    print("log"+str(lineNum)+":abnormal")
                else: #正常，调用模型二
                    if(use_model2=="1"):
                        values = pattern2value[label]
                        vi = now_pattern_index
                        if vi >= window_length and vi < len(values):
                            # Model2 testing
                            seq2 = values[vi - window_length:vi]
                            label2 = values[vi]
                            seq2 = torch.tensor(seq2, dtype=torch.float).view(-1, window_length, len(seq2[0])).to(
                                device)
                            label2 = torch.tensor(label2).view(-1).to(device)
                            mse = 0
                            if label < len(model2) and model2[label] != None:
                                output = model2[label](seq2)
                                # Calculate the MSE of the prediction result and the original result.
                                # If the MSE is within the confidence interval of the Gaussian distribution, the log is a normal log
                                mse = criterion(output[0], label2.to(device))
                            if mse >= mse_threshold: #异常
                                count += 1
                                print("log" + str(lineNum) + ":abnormal")
                                abnormal.append(lineNum)
                pattern_index[label] += 1
                i += 1
    print("time:",time.time()-start_time)
    print("abnormal log number:"+str(count))
    print("abnormal logs:",abnormal)



generate_log_key_and_value()
log_predict(use_model2 = "1")








