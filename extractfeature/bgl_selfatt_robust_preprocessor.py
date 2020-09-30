import io
import re
import os
import random
import math
import json
import pandas as pd
import numpy as np
import torch
import unicodedata


def get_log_template_dic(logparser_event_file):
    dic = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)
    for _, row in data.iterrows():
        dic[row['EventId']] = row['numberID']
    return dic


def generate_train_and_test_file(logparser_structed_file, logparser_event_file, out_dic, train_out_file_name, validation_out_file_name, test_out_file_name, wordvec_path, pattern_vec_out_path, variable_symbol, window_length, step_length):
    log_template_dic = get_log_template_dic(logparser_event_file)
    logparser_result = pd.read_csv(logparser_structed_file, header=0)
    sequences_normal = []
    sequences_abnormal = []
    i = 0
    while i < len(logparser_result) - window_length:
        label = 0
        sequence = []
        for j in range(window_length):
            sequence.append(log_template_dic[logparser_result['EventId'][i+j]])
            if logparser_result['Label'][i+j] != '-':
                label = 1
        if label == 0:
            sequences_normal.append(sequence)
        else:
            sequences_abnormal.append(sequence)
        i += step_length
    random.shuffle(sequences_abnormal)
    random.shuffle(sequences_normal)
    with open(out_dic + train_out_file_name, 'w+') as train_file_obj, open(out_dic + test_out_file_name,
                                                                           'w+') as test_file_obj, open(
            out_dic + validation_out_file_name, 'w+') as validation_file_obj:
        train_file_obj.write('Sequence,label\n')
        test_file_obj.write('Sequence,label\n')
        validation_file_obj.write('Sequence,label\n')
        for i in range(len(sequences_normal)):
            if i < 40000:
                train_file_obj.write(' '.join([str(num_id) for num_id in sequences_normal[i]]))
                train_file_obj.write(', 0\n')
            elif i < 40000 + 50000:
                validation_file_obj.write(' '.join([str(num_id) for num_id in sequences_normal[i]]))
                validation_file_obj.write(', 0\n')
            else:
                test_file_obj.write(' '.join([str(num_id) for num_id in sequences_normal[i]]))
                test_file_obj.write(', 0\n')

        for i in range(len(sequences_abnormal)):
            if i < 40000:
                train_file_obj.write(' '.join([str(num_id) for num_id in sequences_abnormal[i]]))
                train_file_obj.write(', 1\n')
            elif i < 40000 + 5000:
                validation_file_obj.write(' '.join([str(num_id) for num_id in sequences_abnormal[i]]))
                validation_file_obj.write(', 1\n')
            else:
                test_file_obj.write(' '.join([str(num_id) for num_id in sequences_abnormal[i]]))
                test_file_obj.write(', 1\n')

    pattern_to_vec_robust(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol)


def build_pretrained_embeddings(pretrained_file, embedding_dim, id2word, word2id):
    print('embedding matrix loading...')
    vocab_size = len(id2word)
    nn_embeddings = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    wv_file_path = pretrained_file
    count = 0
    pretrain_id_list = []
    with open(wv_file_path, encoding="utf8") as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = unicodedata.normalize('NFD', elems[0])
            if token in word2id:
                count += 1
                word_id = word2id[token]
                nn_embeddings.weight[word_id] = torch.Tensor([float(v) for v in elems[1:]])
                pretrain_id_list.append(word_id)
    embeddings = nn_embeddings.weight.data
    print('embedding matrix loaded.')
    print("#" * 40)
    print("total words in dataset: ", vocab_size)
    print("words in embedding matrix: ", count)
    print("Proportion: ", count / vocab_size * 100, "%")
    print("#" * 40)
    return embeddings, pretrain_id_list


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def get_lower_case_name(text):
    word_list = []
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
    pattern = r'\*|,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    result_list = [x for x in re.split(pattern, log_pattern) if len(x) > 0]
    final_list = list(map(get_lower_case_name, result_list))
    final_list.append(special_list)
    return [x for x in re.split(pattern, final_list.__str__()) if len(x) > 0]


def pattern_to_vec_average(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    data = load_vectors(wordvec_path)
    pattern_to_words = {}
    pattern_to_vectors = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
    print(pattern_to_words)
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if not word in data.keys():
                print('out of 0.1m words', ' ', word)
            else:
                word_used = word_used + 1
                pattern_vector = pattern_vector + np.array(data[word])
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['numberID']] = pattern_to_vectors[row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


def pattern_to_vec_tf_idf_from_log(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    pattern_to_words = {}
    pattern_to_vectors = {}
    pattern_to_occurrences = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    # pattern_num = len(df)
    log_num = 4747963
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
        pattern_to_occurrences[row['EventTemplate'].replace(variable_symbol, '').strip()] = row['Occurrences']
    print(pattern_to_words)
    words_set = set()
    for key in pattern_to_words.keys():
        words_set.update(pattern_to_words[key])
    words_list = list(words_set)
    word2id = {}
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i
        id2word[i] = words_list[i]
    word_embedding, pretrain_id_list = build_pretrained_embeddings(wordvec_path, 300, id2word, word2id)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if word2id[word] in pretrain_id_list:
                word_used = word_used + 1
                weight = wd_list.count(word) / 1.0 / len(pattern_to_words[key])
                if word in IDF.keys():
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + pattern_to_occurrences[key]
                    IDF[word] = math.log10(log_num / 1.0 / pattern_occur_num)
                    # print('tf', weight, 'idf', IDF[word], word)
                    # print(data[word])
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
            else:
                pattern_vector = pattern_vector + np.array(word_embedding[word2id[word]])
                word_used = word_used + 1
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['numberID']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors


def pattern_to_vec_robust(logparser_event_file, wordvec_path, pattern_vec_out_path, variable_symbol):
    pattern_to_words = {}
    pattern_to_vectors = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    df = pd.read_csv(datafile)
    for _, row in df.iterrows():
        wd_list = preprocess_pattern(row['EventTemplate'].replace(variable_symbol, '').strip())
        pattern_to_words[row['EventTemplate'].replace(variable_symbol, '').strip()] = wd_list
    print(pattern_to_words)
    pattern_num = len(pattern_to_words)
    print(pattern_num)
    words_set = set()
    for key in pattern_to_words.keys():
        words_set.update(pattern_to_words[key])
    words_list = list(words_set)
    word2id = {}
    id2word = {}
    for i in range(len(words_list)):
        word2id[words_list[i]] = i
        id2word[i] = words_list[i]
    word_embedding, pretrain_id_list = build_pretrained_embeddings(wordvec_path, 300, id2word, word2id)
    IDF = {}
    for key in pattern_to_words.keys():
        wd_list = pattern_to_words[key]
        pattern_vector = np.array([0.0 for _ in range(300)])
        word_used = 0
        for word in wd_list:
            if word2id[word] in pretrain_id_list:
                word_used = word_used + 1
                weight = wd_list.count(word) / 1.0 / len(pattern_to_words[key])
                if word in IDF.keys():
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
                else:
                    pattern_occur_num = 0
                    for k in pattern_to_words.keys():
                        if word in pattern_to_words[k]:
                            pattern_occur_num = pattern_occur_num + 1
                    IDF[word] = math.log10(pattern_num / 1.0 / pattern_occur_num)
                    # print('tf', weight, 'idf', IDF[word], word)
                    # print(data[word])
                    pattern_vector = pattern_vector + weight * IDF[word] * np.array(word_embedding[word2id[word]])
            else:
                pattern_vector = pattern_vector + np.array(word_embedding[word2id[word]])
                word_used = word_used + 1
        pattern_to_vectors[key] = pattern_vector / word_used
    numberid2vec = {}
    for _, row in df.iterrows():
        numberid2vec[row['numberID']] = pattern_to_vectors[
            row['EventTemplate'].replace(variable_symbol, '').strip()].tolist()
    json_str = json.dumps(numberid2vec)
    with open(pattern_vec_out_path, 'w+') as file_obj:
        file_obj.write(json_str)
    return pattern_to_vectors