#--coding:utf-8--

import os
from enum import Enum

# columns of line
windowSize = 10
# log cluster:1 or sequencer:0
pattern_source = 0

# relation between log_pattern log_key log_line
pattern2log = []
pattern_dic = {}

# log input/output address
log_address = '../sequence/Linux.log'
log_pattern_address_sequencer = '../sequence/linux2.pat'
log_pattern_folder_cluster = '../Logcluster/logcluster/WriteFiles/cluster/'
out_file = '../sequence/Linux_log_vector'


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
                        ifFirst = False
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
                if num_of_line == LineNumber.PATTERN_LINE:
                    pattern = line
                    num_of_line = num_of_line + 1
                elif num_of_line == LineNumber.NUMBERS_LINE:
                    lineNumbers = line.strip().split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = pattern
                    pattern_key = pattern_key + 1
                else:
                    num_of_line = num_of_line + 1


if __name__ == '__main__':
    if pattern_source == 0:
        parse_sequencer()
    else:
        parse_log_cluster()
    with open(out_file, 'x') as out_text:
        with open(log_address, 'rb') as in_log:
            j = 0
            lineNum = 1
            for line in in_log.readlines():
                for i in range(len(pattern2log)):
                    if lineNum in pattern2log[i]:
                        print(i, file=out_text, end='')
                        print(' ', file=out_text, end='')
                        j = j + 1
                        if j == windowSize:
                            print('', file=out_text)
                            j = 0
                        # call method to get value (line, patten_dic[i])
                lineNum = lineNum + 1