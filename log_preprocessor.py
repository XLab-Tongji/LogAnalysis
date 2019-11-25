#--coding:utf-8--

import os
from enum import Enum
import numpy as np
from hashlib import md5
import re
# from GlobalVariables import *

# columns of line
windowSize = 10
# log cluster:1 or sequencer:0
pattern_source = 1

# relation between log_pattern log_key log_line
pattern2log = []
pattern_dic = {}

# log input/output address
## 只要改RootName 即可
RootName='5G'

RootPath='./Data/LogClusterResult-'+RootName+'/'
log_file_dir = './'+RootName+'/'
log_file_name = 'Unino.log'
log_time_file = RootPath+'Vectors/timediff.txt'
log_address = log_file_dir + log_file_name
log_pattern_address_sequencer = './sequence/linux.pat'
log_pattern_folder_cluster = RootPath+'clusters/'
sequencer_out_file = './Data/Vectors1'+log_file_name.split('.')[0]+'_LogKeys_sequencer'
log_cluster_out_file = './Data/Vectors1/'+log_file_name.split('.')[0]+'_LogKeys_logcluster'
log_value_folder = RootPath+'values/'
if pattern_source == 0:
    out_file = sequencer_out_file
else:
    out_file = log_cluster_out_file


# 继承枚举类
class LineNumber(Enum):
    PATTERN_LINE = 0
    NUMBERS_LINE = 3


def parse_sequencer():
    if_first = True
    with open(log_pattern_address_sequencer, 'rb') as in_text:
        log_set = set()
        pattern_key = 0
        last_pattern = ''
        for line in in_text.readlines():
            if (not line.startswith('#'.encode(encoding='utf-8'))) and len(line.strip()):
                if line.startswith('%msgtime%'.encode(encoding='utf-8')):
                    if if_first:
                        last_pattern = line
                        if_first = False
                        continue
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = last_pattern
                    pattern_key = pattern_key + 1
                    log_set = set()
                    last_pattern = line
                else:
                    line = line.decode(encoding='utf-8', errors='strict').strip()
                    lineNumbers = line.split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
    pattern2log.append(log_set)
    pattern_dic[pattern_key] = last_pattern


def parse_log_cluster():
    file_names = os.listdir(log_pattern_folder_cluster)
    pattern_key = 0
    for i in range(len(file_names)):
        with open(log_pattern_folder_cluster + file_names[i], 'r') as in_text:
            num_of_line = 0
            pattern = ''
            log_set = set()
            for line in in_text.readlines():
                if num_of_line == LineNumber.PATTERN_LINE.value:
                    pattern = line
                    num_of_line = num_of_line + 1
                elif num_of_line == LineNumber.NUMBERS_LINE.value:
                    lineNumbers = line.strip().split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = pattern
                    pattern_key = pattern_key + 1
                else:
                    num_of_line = num_of_line + 1

'''
提取value函数
参数tool表示使用的工具 0为sequence 1为logcluster
输出到特定文件
'''
def valueExtract(pattern, log, tool=0, timestamp=None):
    start_char = "%"
    if tool == 1:
        start_char = "*"
    pattern_arr = pattern.split()
    # pattern 写入
    if tool == 0 and not os.path.exists(md5(pattern.encode("utf-8")).hexdigest()+".txt"):
        temp = []
        for pattern_str in pattern_arr:
            if pattern_str[0] == start_char and pattern_str[-1] == start_char:
                temp.append(pattern_str)
        with open("Data/Vectors2/"+md5(pattern.encode("utf-8")).hexdigest()+".txt", "a") as f:
            f.write(", ".join(temp) + "\n")
    # 对于单个日志
    # log_value = [last_timestamp]
    log_value = []
    if timestamp != None:
        log_value.append(timestamp)
    log_arr = log.split()
    log_index = 0
    cur_log_str = log_arr[log_index]
    last_is_pattern = False
    # 遍历模式字符串进行匹配
    for pattern_str in pattern_arr :
        # 如果是value
        # print(pattern_str, cur_log_str)
        if pattern_str[0] == start_char :
        # if pattern_str[0] == start_char and pattern_str[-1] == start_char:
            if pattern_str[1:-1] == "msgtime":
                cur_log_str += (" " + log_arr[log_index+1] + " " + log_arr[log_index+2])
                log_index += 2
                last_timestamp = cur_log_str
            elif pattern_str[1:-1] == "time":
                # time 共有4中情况 目前只能一一判断...
                if (cur_log_str.find("-") == -1 and cur_log_str.find(":") == -1):
                    log_index_add = 0
                    if (log_arr[log_index + 2].find(":") != -1):
                        log_index_add = 2
                    elif (log_arr[log_index + 4].lower() == "est"):
                        log_index_add = 5
                    else:
                        log_index_add = 4
                    for i in range(1, log_index_add + 1):
                        cur_log_str += (" " + log_arr[log_index + i])
                    log_index += log_index_add
            log_len = 1
            if (tool == 1):
                log_len = int(pattern_str[2:pattern_str.find(',')])
            for i in range(1, log_len):
                cur_log_str += (" " + log_arr[log_index+i])
            log_value.append(cur_log_str)
            log_index += log_len
            if (log_index < len(log_arr)):
                cur_log_str = log_arr[log_index]
            last_is_pattern = True
        # 如果是匹配的单词
        elif tool == 1 and pattern_str[0] == '(':
            log_value.append(cur_log_str)
            log_index += 1
            if (log_index < len(log_arr)):
                cur_log_str = log_arr[log_index]
        elif cur_log_str.lower() == pattern_str.lower():
            log_index += 1
            if (log_index < len(log_arr)):
                cur_log_str = log_arr[log_index]
            last_is_pattern = False
        # 如果是单词前一部分匹配
        elif len(cur_log_str) >= len(pattern_str) and cur_log_str.lower()[0:len(pattern_str)] == pattern_str.lower():
            cur_log_str = cur_log_str[len(pattern_str):]
            last_is_pattern = False
        # 此时在前一个字符串中, 如果前一个字符串是value则需要重新取值
        elif last_is_pattern:
            log_index -= 1
            cur_log_str = log_value.pop()
            index = cur_log_str.find(pattern_str)
            log_value.append(cur_log_str[0:index])
            if index+len(pattern_str) == len(cur_log_str):
                log_index += 1
                if (log_index < len(log_arr)):
                    cur_log_str = log_arr[log_index]
            else:
                cur_log_str = cur_log_str[index+len(pattern_str):]
    lines = [", ".join(log_value) + "\n"]
    with open("Data/Vectors1/"+md5(pattern.encode("utf-8")).hexdigest()+".txt", "a+") as f:
        f.writelines(lines)
    return log_value

last_timestamp = "xxx"
def timeExtract(file=None):
    global last_timestamp
    if file == None:
        print("No file input")
        return
    times = []
    with open (log_file_dir + log_file_name) as f:
        for line in f:
            if len(line) > 22:
                times.append(str(timeDiff(line[14:22], last_timestamp)) + "\n")
                last_timestamp = line[14:22]
            else:
                times.append("0\n")
    with open (log_time_file , "w") as f:
        f.writelines(times)

# 时间差
# 暂时使用时分秒做减法
# month_str_num = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
#     "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
def timeDiff(t1, t2):
    if (t2 == "xxx"):
        return 0
    t1_hms_arr = t1.split(":")
    t2_hms_arr = t2.split(":")
    diff_hour = int(t1_hms_arr[0]) - int(t2_hms_arr[0])
    if (diff_hour == -23):
        diff_hour = 1
    diff_min = int(t1_hms_arr[1]) - int(t2_hms_arr[1])
    diff_sec = int(t1_hms_arr[2]) - int(t2_hms_arr[2])
    diff = diff_hour*3600 + diff_min*60 + diff_sec
    return diff

# 向量化
# 改成了对文件向量化
def toVector(pattern, tool=0, file=None):
    # 读取文件内容
    values = []
    with open("Data/Vectors1/"+md5(pattern.encode("utf-8")).hexdigest()+".txt","r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            values.append(line.split(", "))
    new_values = []
    if (tool == 0):
        names = values[0]
        for i in range(1, len(values)):
            value = values[i]
            new_value = [timeDiff(value[1], value[0])]
            for j in range(len(names)):
                if (names[j] == "%integer%" or names[j] == "%float%"):
                    new_value.append(value[j+1])
            new_values.append(new_value)
    else:
        ##------ Start of 新增-------------
        if len(values) <= 0:
            print("no values")
            return 
        val_num = len(values[0])
        isAlldigit = [1]*val_num
        for i in range(val_num):
            for value in values:
                val = value[i]
                if (val.isdigit() or (val[0] == "-" and val[1:].isdigit()) 
                    or re.match(r"-?[0-9]+\.[0-9]+$", val)):
                    #do nothing
                    pass
                else:
                    isAlldigit[i] = 0
        ##------ End of 新增-------------

        for value in values:
            new_value = []
            index=0
            for val in value:
                if isAlldigit[index]:
                    if (val.isdigit() or (val[0] == "-" and val[1:].isdigit()) 
                        or re.match(r"-?[0-9]+\.[0-9]+$", val)):
                        new_value.append(val)
                index+=1
            new_values.append(new_value)
        
    # Normalize
    # print(new_values)
    new_values = np.array(new_values, dtype=float)
    new_values -= np.mean(new_values, axis=0)
    # print(new_values)
    std = np.std(new_values, axis=0)
    # print(std)
    std[std == 0.0] = 1.0
    new_values /= std
    lines = []
    for val in new_values:
        line = str(val[0])
        for i in range(1, len(val)):
            line += ", " + str(val[i])
        lines.append(line + "\n")
    if file == None:
        file = md5(pattern.encode("utf-8")).hexdigest()+"_vector.txt"
    with open(file, "w") as f:
        f.writelines(lines)
    return new_values


if __name__ == '__main__':
    if pattern_source == 0:
        parse_sequencer()
    else:
        parse_log_cluster()
    # out_train=open(RootPath+"logkey_train",'w')  
    # out_test=open(RootPath+"logkey_test",'w')  
    # out_val=open(RootPath+"logkey_val",'w')  
    # out_abnormal=open(RootPath+"logkey_abnormal",'w')  
    # out_text=out_train
    # with open(log_address, 'rb') as in_log:
    #     j = 0
    #     lineNum = 1
    #     for line in in_log.readlines():
    #         for i in range(len(pattern2log)):
    #             if lineNum in pattern2log[i]:
    #                 print(i+1, file=out_text, end='')
    #                 print(' ', file=out_text, end='')
    #                 j = j + 1
    #                 if j == windowSize:
    #                     print('', file=out_text)
    #                     j = 0
    #                 # call method to get value (line, patten_dic[i])
    #         lineNum = lineNum + 1
    #         if lineNum>2000 and lineNum<2500: 
    #             out_text=out_val
    #         if lineNum>2500 and lineNum<3000: 
    #             out_text=out_test
    #         if lineNum>3000 and lineNum<3500: 
    #             out_text=out_abnormal

    # value extract
    with open(log_file_dir + log_file_name) as o_log_file:
        o_log_lines = o_log_file.readlines()
        timeExtract( log_time_file)
        for file in os.listdir(log_pattern_folder_cluster):
            with open(log_pattern_folder_cluster + file) as f:
                with open(log_time_file) as tf:
                    f_lines = f.readlines()
                    tf_lines = tf.readlines()
                    pattern = f_lines[0]
                    o_lines = f_lines[3].split()
                    for o_line in o_lines:
                        valueExtract(pattern, o_log_lines[int(o_line) - 1], 1, tf_lines[int(o_line) - 1].strip('\n'))
                    toVector(pattern, 1, log_value_folder + file)
                    print(file + " done")
