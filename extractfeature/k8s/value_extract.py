#读聚类结果下的7个log 对每一个log 在value.log里找对应的value 再输出到四个部分
import os
from datetime import datetime
import numpy as np

log_file_dir = './extractfeature/k8s/'
RootPath='./Data/LogClusterResult-k8s/'
log_pattern_folder_cluster = RootPath+'clusters/'
file_names = os.listdir(log_pattern_folder_cluster)
if not os.path.exists(RootPath + 'logvalue'):
    os.makedirs(RootPath + 'logvalue')
def get_value():
    with open(log_file_dir + "union.log", 'r')as union:
        with open(RootPath + "logvalue/value.log", 'w+')as v:
            for line in union.readlines():
                valueline = line.split(" ")
                value = ''
                value += str(valueline[0][0:4]) + " " + str(valueline[0][5:7]) + " " + str(valueline[0][8:10]) + " "
                value += str(valueline[1][0:2]) + " " + str(valueline[1][3:5]) + " " + str(valueline[1][6:8]) + " " + str(
                    valueline[1][9:]) + " "
                value += str(valueline[5]) + " "
                value += str(valueline[7][5:7]) + " " + str(valueline[7][13:15]) + "\n"
                v.write(value)

def value_deal():
    with open(RootPath + "logvalue/value.log", "r")as valuedeal:
        values = valuedeal.readlines()

    dealvalue = []
    for i in range(len(values) - 1):
        if (i != 2000 and i != 2500 and i != 3000):  # 把两个模块之间的那个时间差不计算
            v1 = values[i][:-1].split(" ")
            v2 = values[i + 1][:-1].split(" ")
            time1 = datetime(int(v1[0]), int(v1[1]), int(v1[2]), hour=int(v1[3]), minute=int(v1[4]), second=int(v1[5]),
                             microsecond=int(v1[6]))
            time2 = datetime(int(v2[0]), int(v2[1]), int(v2[2]), hour=int(v2[3]), minute=int(v2[4]), second=int(v2[5]),
                             microsecond=int(v2[6]))
            time = (time2.day - time1.day) * 86400000 + (time2.hour - time1.hour) * 3600000 + (
                        time2.minute - time1.minute) * 60000 + (time2.second - time1.second) * 1000 + (
                               time2.microsecond - time1.microsecond)
            v = str(time) + " " + v2[7] + " " + v2[8] + " " + v2[9] + "\n"
            dealvalue.append(v)

    with open(RootPath + "logvalue/dealedvalue.log", "w+")as dvalue:
        for i in dealvalue:
            dvalue.write(i)

    # 做标准化

    with open(RootPath + "logvalue/dealedvalue.log", "r")as v:
        lines = v.readlines()
        # 取出时间
        t = []
        # 取出进程
        process = []
        # 取出端口
        port = []
        # 取出线程
        thread = []
        for i in lines:
            strline = i[:-1].split(" ")  # 去除最后的回车 然后用空格分离
            t.append(int(strline[0]))
            process.append(int(strline[1]))
            port.append(int(strline[2]))
            thread.append(int(strline[3]))

    t_mean = np.mean(t)
    t_std = np.std(t, ddof=1)
    normalize_t = []
    for j in t:
        normalize_t.append((j - t_mean) / t_std)

    normalize_process = []
    process_mean = np.mean(process)
    process_std = np.std(process, ddof=1)
    for j in process:
        normalize_process.append((j - process_mean) / process_std)

    normalize_port = []
    port_mean = np.mean(port)
    port_std = np.std(port, ddof=1)
    for j in port:
        normalize_port.append((j - port_mean) / port_std)

    normalize_thread = []
    thread_mean = np.mean(thread)
    thread_std = np.std(thread, ddof=1)
    for j in thread:
        normalize_thread.append((j - thread_mean) / thread_std)

    # 从四个列表中整合出来新的value列表
    with open(RootPath + "logvalue/normalize_value.log", "w+")as nv:
        for i in range(len(normalize_t)):
            writeline = str(normalize_t[i]) + " " + str(normalize_process[i]) + " " + str(
                normalize_port[i]) + " " + str(normalize_thread[i]) + "\n"
            nv.write(writeline)
def value_extract():
    for i in file_names:
        with open(log_pattern_folder_cluster + i, 'r') as in_text:
            in_text.readline()
            in_text.readline()
            in_text.readline()
            num=in_text.readline()
            num=num[:-1]
            nums=num.split(" ")
            print(nums)
            nums=nums[:-1]
            for j in nums:
                if(j!=" "or j!=''):
                    j=int(j)
                    with open(RootPath + "logvalue/normalize_value.log", 'r')as value:
                        values = value.readlines()
                        if not os.path.exists(RootPath + 'logvalue/logvalue_train'):
                            os.makedirs(RootPath + 'logvalue/logvalue_train')
                            os.makedirs(RootPath + 'logvalue/logvalue_val')
                            os.makedirs(RootPath + 'logvalue/logvalue_test')
                            os.makedirs(RootPath + 'logvalue/logvalue_abnormal')
                        if(j!=1 and j!=2001 and j!=2501 and j!=3001):
                            print(j)
                            if(j<=2000):
                                text = values[j - 1-1]
                                #写到logvalue_train lide 1.log li
                                with open(RootPath + 'logvalue/logvalue_train/'+i,'a+')as t:
                                        t.write(text)
                            elif(j<2500):
                                text = values[j - 2-1]
                                with open(RootPath+'logvalue/logvalue_val/'+i,'a+')as t:
                                        t.write(text)
                            elif(j<3000):
                                text = values[j - 3-1]
                                with open(RootPath + 'logvalue/logvalue_test/' + i, 'a+')as t:
                                    t.write(text)
                            else:
                                text = values[j - 4-1]
                                with open(RootPath + 'logvalue/logvalue_abnormal/' + i, 'a+')as t:
                                    t.write(text)
                        else:
                            continue

