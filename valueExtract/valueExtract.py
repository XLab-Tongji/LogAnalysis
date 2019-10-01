# -*- coding:utf-8 -*-
import numpy as np

'''
提取value函数
参数tool表示使用的工具 0为sequence 1为logcluster
参数output表示输出到的文件
'''
def valueExtract(pattern, logs, tool=0, ouput="result_raw.txt"):
    start_char = "%"
    if tool == 1:
        start_char = "*"
    pattern_arr = pattern.split()
    values = [[]]
    # 第一行value名称
    for pattern_str in pattern_arr:
        if pattern_str[0] == start_char and pattern_str[-1] == start_char:
            values[0].append(pattern_str)
    # 遍历所有日志
    for log in logs:
        log_value = []
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
        values.append(log_value)
    lines = []
    for val in values:
        lines.append(", ".join(val) + "\n")
    with open(ouput, "w") as f:
        f.writelines(lines)
    return values

def toVector(values, ouput="result_vector.txt"):
    names = values[0]
    new_values = []
    for i in range(1, len(values)):
        value = values[i]
        new_value = []
        for j in range(len(names)):
            if (names[j] == r"%integer%" or names[j] == r"%float%"):
                new_value.append(value[j])
        new_values.append(new_value)
    # Normalize
    new_values = np.array(new_values, dtype=float)
    new_values -= np.mean(new_values, axis=0)
    new_values /= np.std(new_values, axis=0)
    lines = []
    for val in new_values:
        line = str(val[0])
        for i in range(1, len(val)):
            line += ", " + str(val[i]);
        lines.append(line + "\n")
    with open(ouput, "w") as f:
        f.writelines(lines)
    return new_values

if __name__ == "__main__":
    logs = []
    with open("input.txt") as f:
        for line in f:
            logs.append(line)
    pattern = logs[0]
    logs = logs[1:]
    values = valueExtract(pattern, logs)
    toVector(values)
    print("=====done=====")