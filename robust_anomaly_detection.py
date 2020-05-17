# -*- coding: UTF-8 -*-

import os
from logparsing.fttree import fttree
from extractfeature import hdfs_ft_preprocessor
from anomalydetection.loganomaly import log_anomaly_sequential_train
from anomalydetection.loganomaly import log_anomaly_sequential_predict
from anomalydetection.robust import bi_lstm_att_train
from anomalydetection.robust import bi_lstm_att_predict
import os 
import re
import numpy as np 
import pandas as pd
from collections import OrderedDict
import json
log_file='./Data/log/hdfs/HDFS_split'
log_file_label='./Data/log/hdfs/HDFS_split_anomaly'

clusters_files='./Data/FTTreeResult-HDFS/clusters/'

temp_directory = './Data/logdeepdata/'
train_file_name = 'robust_log_train.csv'
test_file_name = 'robust_log_test.csv'
valid_file_name = 'robust_log_valid.csv'

# log anomaly sequential model parameters some parameter maybe changed to train similar models
sequence_length = 10
input_size = 300
hidden_size = 128
num_of_layers = 2
# 1 using sigmoid, 2 using softmax
num_of_classes = 1
num_epochs = 50
batch_size = 500
# for robust attention bi
train_root_path = './Data/FTTreeResult-HDFS/robust_att_bi_model_train/'
model_out_path = train_root_path + 'model_out/'
train_file = temp_directory + train_file_name
pattern_vec_json = './Data/logdeepdata/event2semantic_vec.json'



if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)



def generate_train_and_test_file():

    # #将聚类结果读取到clusters_result字典中
    files= os.listdir(clusters_files)
    clusters_result={}
    for file in files: #遍历文件夹
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            f = open(clusters_files+"/"+file); #打开文件
            f.readline()
            linenums=f.readline()
            linenums=linenums.split(" ")
            clusters_result[file]=linenums
    

    with open(log_file,'r')as hdfs_file:
        num=sequence_length
        for i in range(110000):
            log_line=hdfs_file.readline()
            #找这个日志是那个聚类里
            log_key=0
            for key, value in clusters_result.items():
                if str(i) in value:
                    log_key=key
                    break
            if(num>0):
                with open('test10','a+') as t:
                    t.write(str(log_key)+" ")
                num=num-1
            else:
                num=sequence_length
                with open('test10','a+') as t:
                    t.write("\n"+str(log_key)+" ")
                num=num-1



    #生成训练文件


    anamaly_label=''
    with open(log_file_label,'r')as label:
        anamaly_label=label.readline().split()
    train_log=[]
    test_log=[]
    valid_log=[]
    anamaly_num=0
    log_line_num=0
    with open("test10",'r') as f:
        for k in range(2000):
            label=0
            origin_line=f.readline()[:-1]

            for i in anamaly_label:
                i=int(i)
                if(i<sequence_length*(k+1) and i>sequence_length*k ): #0到50
                    label=1
                    anamaly_num+=1
                    break
            a=[]
            a.append(origin_line)
            a.append(label)  
            train_log.append(a)
            # train_log[origin_line]=label
        data_df = pd.DataFrame(data=train_log, columns=['Sequence', 'label'])
        data_df.to_csv("./Data/logdeepdata/robust_log_train.csv",index=None)
        print("训练集中异常有",anamaly_num)

        anamaly_num=0
        for k in range(2000,2500):
            label=0
            origin_line=f.readline()[:-1]

            for i in anamaly_label:
                i=int(i)
                if(i<sequence_length*(k+1) and i>sequence_length*k ): #0到50
                    label=1
                    anamaly_num+=1
                    break
               
            a=[]
            a.append(origin_line)
            a.append(label)  
            valid_log.append(a)
            # train_log[origin_line]=label
        data_df = pd.DataFrame(data=valid_log, columns=['Sequence', 'label'])
        data_df.to_csv("./Data/logdeepdata/robust_log_valid.csv",index=None)
        print("验证集中异常有",anamaly_num)



        anamaly_num=0
        for k in range(2500,11000):
            label=0
            origin_line=f.readline()[:-1]
            for i in anamaly_label:
                i=int(i)
                if(i<sequence_length*(k+1) and i>sequence_length*k ): #0到50
                    label=1
                    anamaly_num+=1
                    break
            a=[]
            a.append(origin_line)
            a.append(label)  
            test_log.append(a)
            # train_log[origin_line]=label
        data_df = pd.DataFrame(data=test_log, columns=['Sequence', 'label'])
        data_df.to_csv("./Data/logdeepdata/robust_log_test.csv",index=None)
        print("测试集中异常有",anamaly_num)
        


    # # 生成json

    with open("./Data/logdeepdata/pattern_vec") as pvec:
        lines=pvec.readlines()
        vecs={}
        for line in lines:
            line=line.split("[:]")
            svec=line[-1][:-1].split()
            vec=[]
            for s in svec:
                vec.append(float(s))
            #找这是第几个pattern

            files= os.listdir(clusters_files)
            clusters_result={}
            for file in files: #遍历文件夹
                if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
                    f = open(clusters_files+"/"+file); #打开文件
                    pattern=f.readline()[:-1]
                    if(pattern==line[0]):
                        vecs[file]=vec
                        break

        event2semvec = json.dumps(vecs)   

        with open('./Data/logdeepdata/event2semantic_vec.json', 'w') as fw:
            fw.write(event2semvec)
                    



def train_model():
    bi_lstm_att_train.train_model(sequence_length, input_size, hidden_size, num_of_layers, num_of_classes, num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_json)


def test_model():
    # do something
    bi_lstm_att_predict.do_predict(input_size, hidden_size, num_of_layers, num_of_classes, sequence_length, model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt', temp_directory + test_file_name, batch_size, pattern_vec_json)

generate_train_and_test_file()
train_model()
test_model()
