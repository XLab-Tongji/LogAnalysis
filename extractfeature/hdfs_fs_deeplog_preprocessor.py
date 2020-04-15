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
    log = log_file_dir+log_file_name
    in_abnormal = log_file_dir+log_file_abnormal_label
    log_value_dir = ['logvalue_train/', 'logvalue_test/']
    log_value_train_directory = log_preprocessor_dir+log_value_dir[0]
    log_value_test_directory = log_preprocessor_dir +log_value_dir[1]

    if not os.path.exists(log_value_train_directory):
        os.makedirs(log_value_train_directory)

    if not os.path.exists(log_value_test_directory):
        os.makedirs(log_value_test_directory)

    log_list = []
    with open(log, 'r') as file:
        content_list = file.readlines()
        log_list = [x.strip() for x in content_list]

    abnormal = get_abnormal(in_abnormal)
    clusters = get_logkey(log_fttree_out_dir)[0]

    num = [0, 170000, 199999]

    for i in range(0, 2):
        for j in range(1, 62):
            print("process:", i, j)
            para1 = []
            para2 = []
            para3 = []
            out_path = log_preprocessor_dir + log_value_dir[i] + str(j) + ".txt"
            for t in clusters[j - 1]:
                s = int(t)
                if (i != 1 and s not in abnormal and s >= num[i] and s < num[i + 1]) or (
                        i == 1 and s >= num[i] and s < num[i + 1]):
                    templog = []
                    for word in log_list[s].split(' '):
                        templog.append(word)
                    para1.append(int(templog[0]))
                    para2.append(int(templog[1]))
                    para3.append(int(templog[2]))
                elif s >= num[i + 1]:
                    break;
            if len(para1) > 0:
                para1 = preprocessing.scale(para1)
            if len(para2) > 0:
                para2 = preprocessing.scale(para2)
            if len(para3) > 0:
                para3 = preprocessing.scale(para3)

            with open(out_path, mode='w', encoding='utf-8') as f:
                for w in range(0, len(para1)):
                    print(para1[w], file=f, end='')
                    print(' ', file=f, end='')
                    print(para2[w], file=f, end='')
                    print(' ', file=f, end='')
                    print(para3[w], file=f, end='')
                    print(' ', file=f, end='')
                    print(' ', file=f)





