# -*- coding:utf-8 -*-
import numpy as np
import os
from hashlib import md5

'''
提取value函数
参数tool表示使用的工具 0为sequence 1为logcluster
输出到特定文件（感觉一条日志一次文件读写8太好）
'''
last_timestamp = "xxx"
def valueExtract(pattern, log, tool=0):
    global last_timestamp
    start_char = "%"
    if tool == 1:
        start_char = "*"
    pattern_arr = pattern.split()
    # pattern 写入
    if not os.path.exists(md5(pattern.encode("utf-8")).hexdigest()+".txt"):
        temp = []
        for pattern_str in pattern_arr:
            if pattern_str[0] == start_char and pattern_str[-1] == start_char:
                temp.append(pattern_str)
        with open(md5(pattern.encode("utf-8")).hexdigest()+".txt", "a") as f:
            f.write(", ".join(temp) + "\n")
    # 对于单个日志
    log_value = [last_timestamp]
    log_arr = log.split()
    log_index = 0
    cur_log_str = log_arr[log_index]
    last_is_pattern = False
    # 遍历模式字符串进行匹配
    for pattern_str in pattern_arr:
        # 如果是value
        if pattern_str[0] == start_char and pattern_str[-1] == start_char:
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
            log_value.append(cur_log_str)
            log_index += 1
            if (log_index < len(log_arr)):
                cur_log_str = log_arr[log_index]
            last_is_pattern = True
        # 如果是匹配的单词
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
    with open(md5(pattern.encode("utf-8")).hexdigest()+".txt", "a") as f:
        f.writelines(lines)
    return log_value

# 时间差
# 暂时使用时分秒做减法
# month_str_num = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
#     "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
def timeDiff(t1, t2):
    if (t2 == "xxx"):
        return 0
    t1_hms_arr = t1.split(" ")[2].split(":")
    t2_hms_arr = t2.split(" ")[2].split(":")
    diff_hour = int(t1_hms_arr[0]) - int(t2_hms_arr[0])
    if (diff_hour == -23):
        diff_hour = 1
    diff_min = int(t1_hms_arr[1]) - int(t2_hms_arr[1])
    diff_sec = int(t1_hms_arr[2]) - int(t2_hms_arr[2])
    diff = diff_hour*3600 + diff_min*60 + diff_sec
    return diff

# 向量化
# 改成了对文件向量化
def toVector(pattern):
    # 读取文件内容
    values = []
    with open(md5(pattern.encode("utf-8")).hexdigest()+".txt") as f:
        for line in f:
            line = line.strip('\n')
            values.append(line.split(", "))
    names = values[0]
    new_values = []
    for i in range(1, len(values)):
        value = values[i]
        new_value = [timeDiff(value[1], value[0])]
        for j in range(len(names)):
            if (names[j] == "%integer%" or names[j] == "%float%"):
                new_value.append(value[j+1])
        new_values.append(new_value)
    # Normalize
    new_values = np.array(new_values, dtype=float)
    new_values -= np.mean(new_values, axis=0)
    std = np.std(new_values, axis=0)
    std[std == 0.0] = 1.0
    new_values /= std
    lines = []
    for val in new_values:
        line = str(val[0])
        for i in range(1, len(val)):
            line += ", " + str(val[i]);
        lines.append(line + "\n")
    with open(md5(pattern.encode("utf-8")).hexdigest()+"_vector.txt", "w") as f:
        f.writelines(lines)
    return new_values

if __name__ == "__main__":
    logs = []
    with open("input.txt") as f:
        for line in f:
            logs.append(line)
    pattern = logs[0]
    logs = logs[1:]
    for log in logs:
        valueExtract(pattern, log)
    toVector(pattern)
    print("=====done=====")