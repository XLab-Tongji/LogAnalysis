#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import math

class Node:
    def __init__(self,_base_pattern,_next_pattern,_next_frequency,_next_pattern3,_next_frequency3):
        self.base_pattern = _base_pattern          # 扫描dataset得到的长度为window_size所有模式
        self.next_pattern = _next_pattern          # 向后扫描一个长度得到的不同模式
        self.next_frequency = _next_frequency      # next_pattern中各模式出现的频数
        self.next_pattern3 = _next_pattern3        # 向后扫描三个长度得到的不同模式
        self.next_frequency3 = _next_frequency3    # next_pattern3中各模式出现的频数

def loadData(infile):
    # 将源文件读入的数据格式化后存储在全局数组
    f = open(infile,'r')
    sourceInLine = f.readlines()
    dataset1 = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split(' ')
        dataset1.append(temp2)
    for i in range(0,len(dataset1)):
        t = len(dataset1[i])-1
        for j in range(t):
            dataset1[i].append(int(dataset1[i][j]))
        del(dataset1[i][0:t+1])
    for i in range(0,len(dataset1)):
        t = len(dataset1[i])
        for j in range(t):
            dataset.append(dataset1[i][j])

def buildTree(window_size,type_num):
    # 构建存储所有模式及其next模式的data_tree
    global data_tree
    global index_table
    data_tree = []
    index_table = dict()
    count = 0
    for i in range(0,len(dataset)-window_size-3):
        index = 0
        for j in range(0,window_size):
            index += dataset[i+j]*pow((type_num+1),(window_size-1-j))
        if index not in index_table:
            index_table[index] = count
            _base_pattern = dataset[i:i+window_size]
            _next_pattern = []
            _next_pattern.append(dataset[i+window_size:i+window_size+1])
            _next_frequency = []
            _next_frequency.append(1)
            _next_pattern3 = []
            _next_pattern3.append(dataset[i+window_size:i+window_size+3])
            _next_frequency3 = []
            _next_frequency3.append(1)
            my_node = Node(_base_pattern,_next_pattern,_next_frequency,_next_pattern3,_next_frequency3)
            data_tree.append(my_node)
            count += 1
        else:
            temp = data_tree[index_table[index]]
            find = 0
            for t in range(0,len(temp.next_pattern)):
                if temp.next_pattern[t] == dataset[i+window_size:i+window_size+1]:
                    find = 1
                    temp.next_frequency[t] += 1
                    break
            if find == 0:
                temp.next_pattern.append(dataset[i+window_size:i+window_size+1])
                temp.next_frequency.append(1)
            find3 = 0
            for t in range(0,len(temp.next_pattern3)):
                if temp.next_pattern3[t] == dataset[i+window_size:i+window_size+3]:
                    find3 = 1
                    temp.next_frequency3[t] += 1
                    break
            if find3 == 0:
                temp.next_pattern3.append(dataset[i+window_size:i+window_size+3])
                temp.next_frequency3.append(1)

    # print(index_table)
    # print(len(index_table))
    # print(data_tree[0].base_pattern)
    # print(data_tree[0].next_pattern)
    # print(data_tree[0].next_frequency)
    # print(data_tree[2].next_pattern3)
    # print(data_tree[2].next_frequency3)

def checkConcurrency(window_size,type_num):
    #对dataset进行并发事件检查，所有并发事件合并为一个新事件
    for i in range(0,len(data_tree)):
        del_index=[]
        concurrency=[]
        for j in range(0,len(data_tree[i].next_pattern3)):
            for k in range(j+1,len(data_tree[i].next_pattern3)):
                if (data_tree[i].next_pattern3[j][0] == data_tree[i].next_pattern3[k][1]) and \
                        (data_tree[i].next_pattern3[j][1] == data_tree[i].next_pattern3[k][0]) and \
                        (data_tree[i].next_pattern3[j][2] == data_tree[i].next_pattern3[k][2]):
                    del_index.append(j)
                    del_index.append(k)
                    concurrency.append([data_tree[i].next_pattern3[j][0], data_tree[i].next_pattern3[j][1]])
                    concurrency.append([data_tree[i].next_pattern3[k][0], data_tree[i].next_pattern3[k][1]])
        del_index.sort(reverse=True)
        # if(len(del_index)>0):
        #     print(del_index)
        #     print(concurrency)
        for t in range(0,len(del_index)):
            del data_tree[i].next_pattern3[del_index[t]]
        for t in range(0,len(dataset)):
            for s in range(0,len(concurrency)):
                if (dataset[t:t+window_size] == data_tree[i].base_pattern) and \
                        (dataset[t+window_size:t+window_size+2] == concurrency[s]) and \
                        (dataset[t+window_size] <= type_num):
                    dataset[t + window_size] = dataset[t + window_size] * 1000 +dataset[t + window_size + 1]
                    dataset[t + window_size + 1] = dataset[t + window_size]
    t = len(dataset)
    for i in range(t-1,0,-1):
        if dataset[i]>1000 and dataset[i]==dataset[i-1]:
            del dataset[i]
    print("并发检查后数据集：",dataset)
    print("dataset长度:", len(dataset))

def checkNewTask(window_size,type_num):
    #对并发检查后的dataset进行新任务检查
    door = 0.02*window_size
    for i in range(0, len(data_tree)):
        new_task = []
        frequency_sum = sum(data_tree[i].next_frequency)
        for j in range(0,len(data_tree[i].next_pattern)):
            if data_tree[i].next_frequency[j]/frequency_sum < door:
                new_task.append(data_tree[i].next_pattern[j][0])
        for t in range(0, len(dataset)-window_size):
            for s in range(0, len(new_task)):
                if (dataset[t:t + window_size] == data_tree[i].base_pattern) and \
                    (dataset[t + window_size] == new_task[s]) :
                    dataset[t+window_size] *= -1
    print("检查新任务后数据集：",dataset)
    print("dataset长度:", len(dataset))

def outputDataset(infile):
    #将进行新任务检查后的dataset输出到格式为'new' + infilename + '.txt'的文件下，并将dataset整理好存储在new_dataset中
    global new_dataset
    new_dataset=[]
    outfile = 'new' + infile + '.txt'
    f=open(outfile,'w')
    start = 0
    end = 0
    for i in range(0,len(dataset)):
        if dataset[i] < 0:
            end = i
            dataset[i]=-1*dataset[i]
            if dataset[start:end+1] not in new_dataset:
                new_dataset.append(dataset[start:end+1])
            start = end + 1
    print("新数据集",new_dataset)
    print("新数据集长度",len(new_dataset))
    for i in range(0,len(new_dataset)):
        for j in range(0,len(new_dataset[i])-1):
            f.write(str(new_dataset[i][j]))
            f.write(' ')
        f.write(str(new_dataset[i][len(new_dataset[i])-1]))
        f.write('\n')

def checkCycle(infile):
    #对new_dataset进行循环检查
    for i in range(0,len(new_dataset)):
        j=0
        while j < len(new_dataset[i]):
            k = 1
            while k<int((len(new_dataset[i])-j)/2):
                 if new_dataset[i][j:j+k]==new_dataset[i][j+k:j+2*k]:
                     m=2
                     new_dataset[i][j] = (new_dataset[i][j] * 1000 + k) * (-1)
                     while (j+(m+1)*k)<=len(new_dataset[i]) and \
                            new_dataset[i][j+(m-1)*k:j+m*k]==new_dataset[i][j+m*k:j+(m+1)*k]:
                        m+=1
                     del new_dataset[i][j+k:j+m*k:1]
                     j+=k-1
                     break
                 k+=1
            j+=1
    print(new_dataset)
    outfile = 'new2' + infile + '.txt'
    f=open(outfile,'w')
    for i in range(0,len(new_dataset)):
        for j in range(0,len(new_dataset[i])-1):
            f.write(str(new_dataset[i][j]))
            f.write(' ')
        f.write(str(new_dataset[i][len(new_dataset[i])-1]))
        f.write('\n')


def mainFlow(infile,window_size,type_num=None):
    # infile：文件名
    # window_size：窗口大小
    # type_num：日志种类数，为可选参数 若未给出则取数据文件种日志序号最大值
    global dataset
    dataset = []
    # 将数据从文件读取到dataset中
    loadData(infile)
    print(dataset)
    print("dataset长度:",len(dataset))
    if type_num is None:
        type_num = max(dataset)
    print("type_num =",type_num)
    # 构建data_tree
    buildTree(window_size,type_num)
    # 检查并发事件
    checkConcurrency(window_size,type_num)                  # 暂时用log1*1000+log2方法进行处理
    buildTree(window_size,type_num)
    # 检查新任务
    checkNewTask(window_size,type_num)                        # 取相反数
    # 输出，重构dataste
    outputDataset(infile)
    # 检查循环事件并输出最终结果
    checkCycle(infile)                                      # -(log*1000+循环单元长度）


mainFlow('vectorize',3,None)
# para1：文件名
# para2：窗口大小
# para3：日志种类数，为可选参数 若未给出则取数据文件种日志序号最大值


