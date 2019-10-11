# coding:utf-8
#!/usr/bin/python

import os
import threading
import re


logclusterTool = os.getcwd() + '/logcluster/logcluster.pl'
inputFileName = raw_input('请输入日志文件名（请将文件放至LogFiles文件夹中）:')
inputFile = os.getcwd() + '/logcluster/LogFiles/' + inputFileName
if '.log' not in inputFile and '.LOG' not in inputFile:
    inputFile = inputFile + '.log'
print("请设置以下几个参数的值：（敲回车直接使用默认值）")
support = raw_input("--support(正整数，默认值为100): ")
wweight = raw_input("--wweight(0～1的小数，默认值为0.5): ")
weightf = raw_input("--weightf(1或者2，默认值为1): ")
wfreq = raw_input("--wfreq(0~1的小数，默认值为0.5): ")
separator = raw_input("--separator(日志信息分隔符，默认为空格，可输入自定义正则表达式)：")
# color = raw_input("--color(低频词高亮颜色，默认为黑色，可输入颜色名，如red、blue)：")
writedump = os.getcwd() + '/logcluster/WriteFiles/dump.txt'
writewords = os.getcwd() + '/logcluster/WriteFiles/words.txt'

if re.match(r'[1-9]+\d*', support) is None:
    support = str(100)
if re.match(r'0(\.\d*)?', wweight) is None:
    wweight = 0.5
if re.match(r'1|2', weightf) is None:
    weightf = 1
if re.match(r'0(\.\d*)?', wfreq) is None:
    wfreq = 0.5

analyzeCommand = 'perl ' + logclusterTool + \
                 ' --input ' + inputFile + \
                 ' --support ' + str(support) + \
                 ' --wweight ' + str(wweight) + \
                 ' --weightf ' + str(weightf) + \
                 ' --wfreq ' + str(wfreq) + \
                 ' --writedump ' + writedump + \
                 ' --writewords ' + writewords + \
                 ' --aggrsup'
if separator != '':
    print('sep')
    analyzeCommand += ' --seperator ' + str(separator)

print(analyzeCommand)
with os.popen(analyzeCommand, 'r') as f:
    text = f.read()
    f.close()
# print(text)
with open(r'result.txt', 'w') as resultf:
    resultf.writelines(text)
    resultf.close()



