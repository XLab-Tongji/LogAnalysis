#!/usr/bin/env python
# -*- coding:utf-8 -*-
from sklearn import preprocessing

log_file = ['./5G_normal_train.log','./5G_normal_val.log','./5G_normal_test.log','./5G_abnormal.log']
log_key_file = ['logkey_train','logkey_test','logkey_val','logkey_abnormal']
log_value_dir = ['logvalue_train/','logvalue_test/','logvalue_val/','logvalue_abnormal/']


log_list = []
with open('./Union.log','r') as file:
    content_list = file.readlines()
    log_list = [x.strip() for x in content_list]
# 汇总
temp1_list = []
temp2_list = []
temp3_list = []
temp4_list = []
temp5_list = []
temp6_list = []
temp7_list = []
temp8_list = []
path = 'LogClusterResult-5G/logvalue_normalize/' + 'all.txt'
for i in range(0, len(log_list)):
    templog = []
    for word in log_list[i].split(' '):
        templog.append(word)
    # 前一条日志信息
    frontlog = []
    if i >= 1:
        for word in log_list[i - 1].split(' '):
            frontlog.append(word)
    temp1 = int(templog[1][0:2])
    temp2 = int(templog[1][3:5])
    if i >= 1:
        temp3 = int(templog[2][0:2]) - int(frontlog[2][0:2])
        temp4 = int(templog[2][3:5]) - int(frontlog[2][3:5])
        temp5 = int(templog[2][6:8]) - int(frontlog[2][6:8])
        temp6 = int(templog[2][9:12]) - int(frontlog[2][9:12])
    else:
        temp3 = 0
        temp4 = 0
        temp5 = 0
        temp6 = 0
    temp7 = 0
    if templog[3][1:16] == '192.168.255.129':
        temp8 = 1
    else:
        temp8 = 2
    try:
        temp7 = int(templog[4], 16)
    except ValueError:
        temp7 = 0
    temp1_list.append(temp1)
    temp2_list.append(temp2)
    temp3_list.append(temp3)
    temp4_list.append(temp4)
    temp5_list.append(temp5)
    temp6_list.append(temp6)
    temp7_list.append(temp7)
    temp8_list.append(temp8)
temp1_list = preprocessing.scale(temp1_list)
temp2_list = preprocessing.scale(temp2_list)
temp3_list = preprocessing.scale(temp3_list)
temp4_list = preprocessing.scale(temp4_list)
temp5_list = preprocessing.scale(temp5_list)
temp6_list = preprocessing.scale(temp6_list)
temp7_list = preprocessing.scale(temp7_list)
temp8_list = preprocessing.scale(temp8_list)
with open(path, mode='w', encoding='utf-8') as f:
    for i in range(0, len(log_list)):
        print(temp1_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp2_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp3_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp4_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp5_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp6_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp7_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(temp8_list[i], file=f, end='')
        print(' ', file=f, end='')
        print(' ', file=f)
for i in range(0,4):
    log_key_list = []
    path = 'LogClusterResult-5G/logkey/'+log_key_file[i]
    with open(path,'r') as log_key:
        for line in log_key:
            temp = line.split(' ')
            for j in range(0,len(temp)-1):
                log_key_list.append(int(temp[j]))
    log_func = []
    for j in range(0,max(log_key_list)):
        temp = []
        log_func.append(temp)
    for j in range(0,len(log_key_list)):
        log_func[log_key_list[j]-1].append(j)
    for j in range(0,max(log_key_list)):
        path = 'LogClusterResult-5G/logvalue_normalize/'+log_value_dir[i] + str(j+1)+'.txt'
        with open(path, mode='w', encoding='utf-8') as f:
            for t in log_func[j]:
                if i==0:
                    index = t
                elif i==1:
                    index = 2000 + t
                elif i==2:
                    index = 2500 + t
                elif i==3:
                    index = 3000 + t
                print("序号:",index)
                print(temp1_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp2_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp3_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp4_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp5_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp6_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp7_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(temp8_list[index], file=f, end='')
                print(' ', file=f, end='')
                print(' ', file=f)










