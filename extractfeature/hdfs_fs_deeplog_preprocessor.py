import os
import linecache
from sklearn import preprocessing

def get_abnormal(in_abnormal):
    abnormal_text = linecache.getline(in_abnormal, 1).strip().split()
    abnormal = []
    for i in range(0,len(abnormal_text)):
        abnormal.append(int(abnormal_text[i]))
    return abnormal

def get_logkey(in_clusters):
    clusters = []
    for i in range(1,62):
        temp = linecache.getline(in_clusters+'/'+str(i), 2).strip()
        clusters.append(temp.split())
    print(clusters[0])
    logkey = []
    for i in range(0,199999):
        logkey.append(0)
    for i in range(0,61):
        for j in range(0,len(clusters[i])):
            logkey[int(clusters[i][j])-1] = i+1
    return clusters,logkey

# 将原日志文件分成训练集和测试集两部分,并生成测试集label
def log_split(log_file_dir,log_file_name,log_file_abnormal_label,log_preprocessor_dir):
    log = log_file_dir+log_file_name
    in_abnormal = log_file_dir+log_file_abnormal_label
    abnormal = get_abnormal(in_abnormal)
    log_train = open(log_preprocessor_dir + "HDFS_train.log", 'w')
    log_test = open(log_preprocessor_dir + "HDFS_test.log", 'w')
    with open(log,'r') as f:
        text = f.readlines()
        print(len(text))
        for i in range(0,170000):
            if i not in abnormal:
                print(text[i], file=log_train,end='')
        for i in range(170000,199999):
            print(text[i], file=log_test,end='')
    # label
    abnormal_label = []
    with open(in_abnormal, 'r') as file:
        line = file.readline()
        label = line.split()
        for i in range(0, len(label)):
            if int(label[i]) > 170000:
                abnormal_label.append(int(label[i]) - 170000)
    with open(log_preprocessor_dir+"HDFS_abnormal_label.txt", 'w') as file:
        for i in range(0, len(abnormal_label)):
            print(abnormal_label[i], file=file, end='')
            print(' ', file=file, end='')

# 生成log_key
def generate_log_key(log_file_dir,log_file_abnormal_label,log_preprocessor_dir,log_fttree_out_dir):
    log_key_directory = log_preprocessor_dir+"logkey"
    in_abnormal = log_file_dir + log_file_abnormal_label
    if not os.path.exists(log_key_directory):
        os.makedirs(log_key_directory)
    abnormal = get_abnormal(in_abnormal)
    logkey = get_logkey(log_fttree_out_dir)[1]
    out_train = open(log_key_directory + "/logkey_train", 'w')
    out_test = open(log_key_directory + "/logkey_test", 'w')
    for i in range(0,170000):
        if i not in abnormal:
            print(logkey[i], file=out_train, end='')
            print(' ', file=out_train, end='')
    for i in range(170000,199999):
        print(logkey[i], file=out_test, end='')
        print(' ', file=out_test, end='')

# 提取并处理log_value
def generate_log_value(log_file_dir,log_file_name,log_file_abnormal_label,log_preprocessor_dir,log_fttree_out_dir):
    N_CLUSTER = 21
    WORD2VEC_FILE = 'word2vec'
    STRING_VECTOR_FILE = 'string_vector'

    log_list = []
    word_vector = {}

    # log = log_file_dir+log_file_name
    word2vec = log_file_dir+WORD2VEC_FILE
    string_vector = log_file_dir+STRING_VECTOR_FILE
    in_abnormal = log_file_dir+log_file_abnormal_label

    log_value_dir = ['logvalue_train/', 'logvalue_test/']
    log_value_train_directory = log_preprocessor_dir+log_value_dir[0]
    log_value_test_directory = log_preprocessor_dir +log_value_dir[1]

    if not os.path.exists(log_value_train_directory):
        os.makedirs(log_value_train_directory)

    if not os.path.exists(log_value_test_directory):
        os.makedirs(log_value_test_directory)

    with open(string_vector, 'r') as file:
        content_list = file.readlines()
        log_list = [x.strip() for x in content_list]

    with open(word2vec, 'r') as r:
        for line in r.readlines():
            list_line = line.split(' ')
            value = list(map(float, list_line[1:]))
            key = list_line[0]
            word_vector[key] = value

    abnormal = get_abnormal(in_abnormal)
    clusters = get_logkey(log_fttree_out_dir)[0]

    num = [0, 170000, 199999]

    for i in range(len(log_value_dir)):
        for j in range(N_CLUSTER):
            print("process:", i, j)
            out_path = log_preprocessor_dir + log_value_dir[i] + str(j+1) + ".txt"
            write_list = []
            for t in clusters[j]:
                s = int(t)
                if (i != 1 and s not in abnormal and num[i] <= s < num[i + 1]) or (
                        i == 1 and num[i] <= s < num[i + 1]):
                    output = calc_sentence_vector(log_list[s],word_vector)
                    write_list.append(output)
                elif s >= num[i + 1]:
                    break

            with open(out_path, mode='w', encoding='utf-8') as f:
                f.write('\n'.join(write_list))

def calc_sentence_vector(sentence,word_vector):
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
        for i,j in zip(old_vector,another_vector):
            new_vector.append(i+j)
        old_vector = new_vector

    word_count = len(words)
    for idx,value in enumerate(old_vector):
        old_vector[idx] = value/word_count
    vector_str = list(map(str, old_vector))
    output = ','.join(vector_str)
    return output
