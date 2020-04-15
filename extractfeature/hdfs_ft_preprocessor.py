import os
import io
import re
import numpy as np
import math
# 把异常的删除就是正常的 暂时
# 选取没有异常的块
# cluster里读出来有正常的 也有异常的 根据anomaly csv 区别

#!!!!还需要 分词 对fttree的结果处理 很多不是单词

special_patterns = {'dfs.FSNamesystem:': ['dfs', 'FS', 'Name', 'system'], 'dfs.FSDataset:': ['dfs', 'FS', 'dataset']}

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def get_lower_case_name(text):
    word_list = []
    if text in special_patterns:
        return
    for index, char in enumerate(text):
        if not char.isupper():
            break
        else:
            if index == len(text) - 1:
                return [text]
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            word_list.append("".join(lst))
            lst = []
        lst.append(char)
    word_list.append("".join(lst))
    return word_list


def preprocess_pattern(log_pattern):
    special_list = []
    if log_pattern.split(' ')[0] in special_patterns.keys():
        special_list = special_patterns[log_pattern.split(' ')[0]]
        log_pattern = log_pattern[len(log_pattern.split(' ')[0]):]
    pattern = r'\*|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = [x for x in re.split(pattern, log_pattern) if len(x) > 0]
    final_list = list(map(get_lower_case_name, result_list))
    final_list.append(special_list)
    return [x for x in re.split(pattern, final_list.__str__()) if len(x) > 0]


def pattern_to_vec(cluster_directory, wordvec_path, pattern_vec_out_path):
    data = load_vectors(wordvec_path)
    pattern_to_words = {}
    pattern_to_vectors = {}
    file_names = os.listdir(cluster_directory)
    pattern_num = len(file_names)
    for file_name in file_names:
        with open(cluster_directory + file_name, 'r') as cluster:
            lines = cluster.readlines()
            wd_list = preprocess_pattern(lines[0].strip())
            pattern_to_words[lines[0].strip()] = wd_list
    print(pattern_to_words)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if not word in data.keys():
                print('out of 0.1m words', ' ', word)
            else:
                word_used = word_used + 1
                weight = wd_list.count(word)/1.0/len(pattern_to_words[key])
                if word in IDF.keys():
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(data[word])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + 1
                    IDF[word] = math.log10(pattern_num/1.0/pattern_occur_num)
                    #print('tf', weight, 'idf', IDF[word], word)
                    #print(data[word])
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(data[word])
        pattern_to_vectors[key] = pattern_vector / word_used
    with open(pattern_vec_out_path, 'w+') as file_obj:
        for key in pattern_to_vectors.keys():
            file_obj.write(key)
            file_obj.write('[:]')
            for f in pattern_to_vectors[key]:
                file_obj.write(str(f))
                file_obj.write(' ')
            file_obj.write('\n')
    return pattern_to_vectors


def preprocessor_hdfs_ft(cluster_directory, anomaly_file_path, wordvec_path, out_dic, train_out_file_name, test_out_file_name, label_out_file_name, pattern_vec_out_path, degree, num_of_lines):
    anomaly_log_lines = set()
    with open(anomaly_file_path, 'r') as anomaly_file:
        line = anomaly_file.readline()
        lines_str = line.split(' ')
        anomaly_log_lines.update([int(x) for x in lines_str if len(x) > 0])

    pattern_vec = pattern_to_vec(cluster_directory, wordvec_path, pattern_vec_out_path)

    log_cluster = {}
    file_names = os.listdir(cluster_directory)
    for file_name in file_names:
        with open(cluster_directory + file_name, 'r') as cluster:
            lines = cluster.readlines()
            line_numbers = [int(x) for x in lines[1].split(' ') if len(x) > 0]
            for number in line_numbers:
                if not (number in anomaly_log_lines and number < int(degree*num_of_lines)):
                    log_cluster[number] = pattern_vec[lines[0].strip()]

    with open(out_dic + train_out_file_name, 'w+') as train_file_obj, open(out_dic + test_out_file_name, 'w+') as test_file_obj, open(out_dic + label_out_file_name, 'w+') as label_file_obj:
        count = 1
        for i in sorted(log_cluster):
            if i < int(degree*num_of_lines):
                for f in log_cluster[i]:
                    train_file_obj.write(str(f))
                    train_file_obj.write(' ')
                if count % 200 == 0:
                    train_file_obj.write('\n')
                else:
                    train_file_obj.write(', ')
                count = count + 1
            else:
                if i == int(degree*num_of_lines):
                    count = 1
                if i in anomaly_log_lines:
                    label_file_obj.write(str(count))
                    label_file_obj.write(' ')
                for f in log_cluster[i]:
                    test_file_obj.write(str(f))
                    test_file_obj.write(' ')
                if count % 200 == 0:
                    test_file_obj.write('\n')
                else:
                    test_file_obj.write(', ')
                count = count + 1