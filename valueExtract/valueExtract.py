'''
提取value函数
参数tool表示使用的工具 0为sequence 1为logcluster
参数output表示输出到的文件
'''
def valueExtract(pattern, logs, tool=0, ouput="result.txt"):
    start_char = "%"
    if tool == 1:
        start_char = "*"
    pattern_arr = pattern.split()
    values = []
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

'''
向量化目前还没做，主要是不确定我的理解是否正确。
先记录一下我的理解
举例来说，如果得到的value数组如下
[t1, v11, v21]
[t2, v12, v22]
[t3, v13, v23]
......
[tn, v1n, v2n]

我的理解是，首先把时间变为t1-t0，t2-t1这种形式
然后把每个value都进行标准化（即将每一列标准化）（应该只针对定量的数据）
[t1-t0, v11', v21']
[t2-t1, v12', v22']
......
[tn-t(n-1), v1n', v2n']
然后定一个时间步长，每个time step的输入是上面的一个向量（相当于有多个feature）
也就是LSTM输入的x1...xt中每个x都是一个向量（拥有多个feature）

string如何处理暂时不清楚
'''

if __name__ == "__main__":
    logs = []
    with open("input.txt") as f:
        for line in f:
            logs.append(line)
    pattern = logs[0]
    logs = logs[1:]
    valueExtract(pattern, logs)
    print("=====done=====")