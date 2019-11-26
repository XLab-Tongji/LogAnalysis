#!/usr/bin/env python 
# -*- coding:utf-8 -*-

log_file = ['./5G_normal_train.log','./5G_normal_val.log','./5G_normal_test.log','./5G_abnormal.log']
log_key_file = ['logkey_train','logkey_test','logkey_val','logkey_abnormal']
log_value_dir = ['logvalue_train/','logvalue_test/','logvalue_val/','logvalue_abnormal/']


for i in range(0,4):
    log_key_list = []
    path = 'LogClusterResult-5G/logkey/'+log_key_file[i]
    with open(path,'r') as log_key:
        for line in log_key:
            temp = line.split(' ')
            for j in range(0,len(temp)-1):
                log_key_list.append(int(temp[j]))
    print(log_key_list)
    print(len(log_key_list))
    print(max(log_key_list))
    log_list = []
    with open(log_file[i],'r') as file:
        content_list = file.readlines()
        log_list = [x.strip() for x in content_list]
    log_func = []
    for j in range(0,max(log_key_list)):
        temp = []
        log_func.append(temp)
    for j in range(0,len(log_key_list)):
        log_func[log_key_list[j]-1].append(j)
    print(len(log_func[13]))
    for j in range(0,max(log_key_list)):
        path = 'LogClusterResult-5G/logvalue/'+log_value_dir[i] + str(j+1)+'.txt'
        with open(path, mode='w', encoding='utf-8') as f:
            for t in log_func[j]:
                templog = []
                for word in log_list[t].split(' '):
                    templog.append(word)
                temp1 = int(templog[1][0:2])
                temp2 = int(templog[1][3:5])
                temp3 = int(templog[2][0:2])
                temp4 = int(templog[2][3:5])
                temp5 = int(templog[2][6:8])
                temp6 = int(templog[2][9:12])
                temp7 = 0
                temp8 = int(templog[3][1:4])
                temp9 = int(templog[3][5:8])
                temp10 = int(templog[3][9:12])
                temp11 = int(templog[3][13:16])
                try:
                    temp7 = int(templog[4],16)
                except ValueError:
                    temp7 = 0
                print(t)
                print(temp1,file=f,end='')
                print(' ', file=f, end='')
                print(temp2,file=f,end='')
                print(' ', file=f, end='')
                print(temp3,file=f,end='')
                print(' ', file=f, end='')
                print(temp4,file=f,end='')
                print(' ', file=f, end='')
                print(temp5,file=f,end='')
                print(' ', file=f, end='')
                print(temp6,file=f,end='')
                print(' ', file=f, end='')
                print(temp7,file=f,end='')
                print(' ', file=f, end='')
                print(temp8,file=f,end='')
                print(' ', file=f, end='')
                print(temp9,file=f,end='')
                print(' ', file=f, end='')
                print(temp10,file=f,end='')
                print(' ', file=f, end='')
                print(temp11,file=f,end='')
                print(' ', file=f, end='')
                print(' ', file=f)






