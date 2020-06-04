import sys
path = "C:\\study\\code\\LogAnalysis\\"
sys.path.append(path)
import os
from logparsing.fttree import fttree
import linecache
from anomalydetection.deeplog import log_predict

base_dir = path+'/java/'
# log_detect
log_detect_dir = base_dir+'/detect_log/'
log_detect = "HDFS_detect.log"
# log_detect = sys.argv[1]
# FT-tree
train_fttree_out_dir = path+'Data/FTTreeResult-HDFS/clusters/'
detect_fttree_out_dir = log_detect_dir +'detect_clusters/'
new_fttree_out_dir = log_detect_dir +'clusters/'
# log_train,log_test,logkey,logvalue
log_value_detect = log_detect_dir +'logvalue_test/'
WORD2VEC_FILE ='C:\study\code\LogAnalysis\Data\log\hdfs\word2vec'
STRING_VECTOR_FILE ='C:\study\code\LogAnalysis\Data\log\hdfs\string_vector'
# model
model_dir =  path+'Data/FTTreeResult-HDFS/'+'deeplog_model_train/'
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
use_model2 = True


if not os.path.exists(detect_fttree_out_dir ):
    os.makedirs(detect_fttree_out_dir)
if not os.path.exists(new_fttree_out_dir ):
    os.makedirs(new_fttree_out_dir)

def pattern_extract():
    fttree.pattern_extract(log_detect_dir, log_detect, detect_fttree_out_dir, 5, 4, 2)

def generate_log_key_and_value():
    clusters_num = len(os.listdir(detect_fttree_out_dir))-1
    print(clusters_num)
    train_clusters_num = len(os.listdir(train_fttree_out_dir))-1
    print(train_clusters_num)
    clusters = []
    for i in range(1,clusters_num+1):
        temp = linecache.getline(detect_fttree_out_dir+str(i), 2).strip()
        clusters.append(temp.split())
    logkey = []
    detect_file = open(log_detect_dir+log_detect,'r')
    detect_log_num = len(detect_file.readlines())
    for i in range(0,detect_log_num):
        logkey.append(0)
    for i in range(0,clusters_num):
        for j in range(0,len(clusters[i])):
            logkey[int(clusters[i][j])] = i+1
    #print(logkey)
    detect_key_pattern = []
    with open(detect_fttree_out_dir+'key_pattern.txt','r') as file:
        for line in file.readlines():
            detect_key_pattern.append(line.split('\n')[0])
    train_key_pattern = []
    with open(train_fttree_out_dir+'key_pattern.txt','r') as file:
        for line in file.readlines():
            train_key_pattern.append(line.split('\n')[0])
    #print(detect_key_pattern)
    #print(train_key_pattern)

    detect_key_to_train_key = {}
    detect_key_to_template = {}
    for i in range(0,len(train_key_pattern)):
        detect_key_to_template[i+1] = train_key_pattern[i]
    for i in range(0,len(detect_key_pattern)):
        if detect_key_pattern[i] in train_key_pattern:
            index = train_key_pattern.index(detect_key_pattern[i])
            detect_key_to_train_key[i+1]=index+1
            detect_key_to_template[i+1]=detect_key_pattern[i]
        else:
            detect_key_to_train_key[i + 1]=train_clusters_num +1
            detect_key_to_template[train_clusters_num +1]=detect_key_pattern[i]
            train_clusters_num += 1
    #(detect_key_to_train_key)

    with open(log_detect_dir + 'clusters/key_pattern.txt', 'w') as file:
        for i in range(1,len(detect_key_to_template)+1):
            file.write(detect_key_to_template[i]+'\n')

    new_logkey = []
    with open(log_detect_dir+'detect_logkey.txt','w') as file:
        for key in logkey:
            new_logkey.append(detect_key_to_train_key[key])
            file.write(str(detect_key_to_train_key[key])+" ")
    #print("new_logkey:",new_logkey)

    N_CLUSTER = max(new_logkey)
    print("N_CLUSTER:",N_CLUSTER)
    new_clusters = []
    for i in range(0,N_CLUSTER):
        new_clusters.append([])
    for i in range(0,len(clusters)):
        new_clusters[detect_key_to_train_key[i+1]-1] = clusters[i]
    for i in range(0,len(new_clusters)):
        with open(new_fttree_out_dir+str(i+1)+'.txt','w') as f:
            for j in range(0,len(new_clusters[i])):
                f.write(new_clusters[i][j]+" ")

    #print("new_clusters:",new_clusters)

    log_list = []
    word_vector = {}
    word2vec = WORD2VEC_FILE
    string_vector = STRING_VECTOR_FILE
    if not os.path.exists(log_value_detect):
        os.makedirs(log_value_detect)
    with open(string_vector, 'r') as file:
        content_list = file.readlines()
        log_list = [x.strip() for x in content_list]
    with open(word2vec, 'r') as r:
        for line in r.readlines():
            list_line = line.split(' ')
            value = list(map(float, list_line[1:]))
            key = list_line[0]
            word_vector[key] = value
    for j in range(N_CLUSTER):
        # print("process:", j)
        out_path = log_value_detect+ str(j + 1) + ".txt"
        write_list = []
        for t in new_clusters[j]:
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
        # print(word)
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

def test_model():
    model1_name = 'Adam_batch_size=' + str(model1_batch_size) + ';epoch=' + str(model1_num_epochs) + '.pt'
    log_predict.do_predict(log_detect_dir,log_fttree_out_dir,model_dir,model1_name,model2_num_epochs,window_length, input_size, hidden_size, num_of_layers, num_candidates, mse_threshold, use_model2)

print("start preprocessing:")
# pattern_extract()
generate_log_key_and_value()
print("loading model and predict:")
#test_model()








