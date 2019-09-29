#--coding:utf-8--

# columns of line
import os

windowSize = 10
# logcluster:1 or sequencer:0
pattern_source = 0

# relation between logpattern logkey logline
pattern2log = []
pattern_dic = {}


def parse_sequencer():
    ifFirst = True
    with open('../sequence/linux2.pat', 'rb') as in_text:
        log_set = set()
        pattern_key = 0
        last_pattern = ''
        for line in in_text.readlines():
            if (not line.startswith('#'.encode(encoding='utf-8'))) and len(line.strip()):
                if line.startswith('%msgtime%'.encode(encoding='utf-8')):
                    if ifFirst:
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
    file_names = os.listdir('../Logcluster/logcluster/WriteFiles/cluster')
    pattern_key = 0
    for i in range(len(file_names)):
        with open('../Logcluster/logcluster/WriteFiles/cluster/' + file_names[i], 'r') as in_text:
            num = 0
            pattern = ''
            log_set = set()
            for line in in_text.readlines():
                if num == 0:
                    pattern = line
                    num = num + 1
                elif num == 3:
                    lineNumbers = line.strip().split(' ')
                    lineNumbers = [int(x) for x in lineNumbers]
                    for x in lineNumbers:
                        log_set.add(x)
                    pattern2log.append(log_set)
                    pattern_dic[pattern_key] = pattern
                    pattern_key = pattern_key + 1
                else:
                    num = num + 1



if __name__ == '__main__':
    if pattern_source == 0:
        parse_sequencer()
    else:
        parse_log_cluster()
    with open('../sequence/Linux_log_vector', 'x') as out_text:
        with open('../sequence/Linux.log', 'rb') as in_log:
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