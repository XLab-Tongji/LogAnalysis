import csv
import os
import random

class hdfs_deeplog_preprocessor:
    # 日志变量设置
    LOG_LINE = 400000
    NUM_OF_LOGKEY = 26
    VECTOR_DIMENSION = 10
    NORMAL_STAGE_TO_STAGE_SIZE = [2000, 1000, 1000]
    ABNORMAL_STAGE_TO_STAGE_SIZE = [800, 200, 200]

    # 读入数据部分
    ANOMALY_LABEL = '../Data/log/hdfs/anomaly_label.csv'
    LOG_FILE = '../Data/log/hdfs/HDFS_40w'
    MOFIFIED_LOG_FILE = '../Data/log/hdfs/modified_HDFS_40w'
    WORD_VECTOR_FILE = '../Data/log/hdfs/word2vec_HDFS_40w'
    LOGKEY_DIR = '../Data/FTTreeResult-HDFS/clusters/'
    is_block_normal = {}
    block_to_lines = {}
    line_to_logkey = []
    word_to_vector = {}
    modified_logs = []

    # 输出数据部分
    OUTPUT_DIR_PREFIX = '../Data/log_preprocessor/'
    STAGE_TO_OUTPUT_DIR_INFIX = ['train/','validate/','test/']
    normal_blocks = []
    abnormal_blocks = []
    normal_block_index_to_stage = []
    abnormal_block_index_to_stage = []



    '''
    -----------------------------------------------
    以下是load_data部分
    -----------------------------------------------
    '''

    def load_normal_info(self):
        NORMAL_WORD = 'Normal'
        FIRST_LINE_BLOCK_NAME = 'BlockId'

        with open(self.ANOMALY_LABEL,'r') as f:
            lines = csv.reader(f)
            for line in lines:
                block = line[0]
                normal_word = line[1]
                if normal_word == NORMAL_WORD:
                    normal_info = True
                else:
                    normal_info = False
                if block != FIRST_LINE_BLOCK_NAME:
                    self.is_block_normal[block] = normal_info

    def load_line_info(self):
        with open(self.LOG_FILE,'r') as f:
            for line_index in range(self.LOG_LINE):
                line = f.readline()
                block = self.get_blockid(line)
                if block not in self.block_to_lines.keys():
                    self.block_to_lines[block] = []
                self.block_to_lines[block].append(line_index)
        # print(self.block_to_lines['blk_-1608999687919862906'])

    def load_logkey_info(self):
        self.line_to_logkey = [0 for i in range(self.LOG_LINE)]
        for logkey in range(1,self.NUM_OF_LOGKEY+1):
            with open(self.LOGKEY_DIR+str(logkey),'r') as f:
                f.readline()
                lines = f.readline().strip().split(' ')
                for line in lines:
                    line_index = int(line)
                    if line_index>=self.LOG_LINE:
                        print('cluster文件中某行的行数过大')
                        print(line)
                        exit(2)
                    self.line_to_logkey[line_index] = logkey

    def load_word_vector(self):
        with open(self.WORD_VECTOR_FILE, 'r') as r:
            for line in r.readlines():
                list_line = line.split(' ')
                value = list(map(float, list_line[1:]))
                key = list_line[0]
                self.word_to_vector[key] = value

    def load_modified_log(self):
        with open(self.MOFIFIED_LOG_FILE, 'r') as file:
            content_list = file.readlines()
            self.modified_logs = [x.strip() for x in content_list]

    def generate_block_list(self):
        for block in self.block_to_lines.keys():
            if self.is_block_normal[block]:
                self.normal_blocks.append(block)
            else:
                self.abnormal_blocks.append(block)

    '''
    -----------------------------------------------
    以下是一些辅助函数
    -----------------------------------------------
    '''

    def get_blockid(self, line):
        words = line.strip().split(' ')
        for word in words:
            if len(word)>4 and word[:4] == 'blk_':
                return word
        print('无法找到block_id')
        print(line)
        exit(1)


    def get_sentence_vector(self, sentence):
        words = sentence.split(' ')
        old_vector = [0.0 for i in range(self.VECTOR_DIMENSION)]
        for word in words:
            # print(word)
            if word not in self.word_to_vector.keys():
                another_vector = [0.0 for i in range(self.VECTOR_DIMENSION)]
            else:
                another_vector = self.word_to_vector[word]
            new_vector = []
            for i, j in zip(old_vector, another_vector):
                new_vector.append(i + j)
            old_vector = new_vector

        word_count = len(words)
        for idx, value in enumerate(old_vector):
            old_vector[idx] = value / word_count
        vector_str = list(map(str, old_vector))
        sentence_vector = ','.join(vector_str)
        return sentence_vector

    def get_logkey_and_logvalue_for_session(self, lines):
        logkeys = []
        logkey_to_logvalues = [[] for i in range(self.NUM_OF_LOGKEY+1)]
        for line in lines:
            logkey = self.line_to_logkey[line]
            logkeys.append(logkey)
            log = self.modified_logs[line]
            vector = self.get_sentence_vector(log)
            logkey_to_logvalues[logkey].append(vector)
        return logkeys,logkey_to_logvalues
    '''
    -----------------------------------------------
    以下是output_logkey_and_logvalue部分
    -----------------------------------------------
    '''

    def get_block_stage_info(self,total_length,stage_to_length):
        if sum(stage_to_length) > total_length:
            print('要输出的条目太大，大于数据集中存在的条目。')
            print(total_length)
            print(stage_to_length)
            exit(3)
        table = [-1 for i in range(total_length)]
        for stage in range(len(stage_to_length)):
            stage_length_count = 0
            while stage_length_count < stage_to_length[stage]:
                table_index = random.randint(0,total_length-1)
                if table[table_index] == -1:
                    table[table_index] = stage
                    stage_length_count += 1
        return table

    def output(self,stage,output_normal):
        if output_normal:
            OUTPUT_DIR_SUFFIXES = ['logkey/','logvalue/normal/']
            LOGKEY_FILE = 'normal'
            blocks = self.normal_blocks
            block_index_to_stage = self.normal_block_index_to_stage
        else:
            OUTPUT_DIR_SUFFIXES = ['logkey/', 'logvalue/abnormal/']
            LOGKEY_FILE = 'abnormal'
            blocks = self.abnormal_blocks
            block_index_to_stage = self.abnormal_block_index_to_stage

        LOGKEY_OUTPUT_DIR = self.OUTPUT_DIR_PREFIX + \
                            self.STAGE_TO_OUTPUT_DIR_INFIX[stage] + OUTPUT_DIR_SUFFIXES[0]
        LOGVALUE_OUTPUT_DIR = self.OUTPUT_DIR_PREFIX + \
                              self.STAGE_TO_OUTPUT_DIR_INFIX[stage] + OUTPUT_DIR_SUFFIXES[1]
        if not os.path.exists(LOGKEY_OUTPUT_DIR):
                os.makedirs(LOGKEY_OUTPUT_DIR)
        if not os.path.exists(LOGVALUE_OUTPUT_DIR):
                os.makedirs(LOGVALUE_OUTPUT_DIR)
        logkey_writelist = []
        logkey_to_logvalue_writelist = [[] for i in range(self.NUM_OF_LOGKEY+1)]

        for block_index,block in enumerate(blocks):
            if block_index_to_stage[block_index] == stage:
                lines = self.block_to_lines[block]
                logkeys, logkey_to_logvalues = \
                    self.get_logkey_and_logvalue_for_session(lines)
                logkey_line = ' '.join(str(logkey) for logkey in logkeys)
                logkey_writelist.append(logkey_line+'\n')
                for logkey in range(1,self.NUM_OF_LOGKEY+1):
                    if len(logkey_to_logvalues[logkey]) == 0:
                        logvalue_line = '-1'
                    else:
                        logvalue_line = ' '.join(logkey_to_logvalues[logkey])
                    logkey_to_logvalue_writelist[logkey].append(logvalue_line+'\n')

        with open(LOGKEY_OUTPUT_DIR + LOGKEY_FILE,'w') as f:
            f.writelines(logkey_writelist)
        for logkey in range(1,self.NUM_OF_LOGKEY+1):
            LOGVALUE_FILE = str(logkey)
            with open(LOGVALUE_OUTPUT_DIR + LOGVALUE_FILE,'w') as f:
                f.writelines(logkey_to_logvalue_writelist[logkey])


    '''
    -----------------------------------------------
    以下是main函数部分
    -----------------------------------------------
    '''


    def load_data(self):
        self.load_normal_info()
        print('正常/异常标签加载成功')
        self.load_line_info()
        print('数据集block信息加载成功')
        self.load_logkey_info()
        print('从clusters取出logkey信息成功')
        self.load_word_vector()
        print('读入word vector信息成功')
        self.load_modified_log()
        print('读入log信息成功')
        self.generate_block_list()
        print('将block划分为正常/异常成功')

    def output_logkey_and_logvalue(self):
        self.abnormal_block_index_to_stage = self.get_block_stage_info \
            (len(self.abnormal_blocks),self.ABNORMAL_STAGE_TO_STAGE_SIZE)
        print('给异常block选择train validate test数据成功')
        self.normal_block_index_to_stage = self.get_block_stage_info \
            (len(self.normal_blocks), self.NORMAL_STAGE_TO_STAGE_SIZE)
        print('给正常block选择train validate test数据成功')
        for stage in range(len(self.STAGE_TO_OUTPUT_DIR_INFIX)):
            self.output(stage, output_normal=True)
            print('给阶段' + str(stage) + '输出正常logkey和logvalue成功')
            self.output(stage, output_normal=False)
            print('给阶段' + str(stage) + '输出异常logkey和logvalue成功')

    def __init__(self):
        self.load_data()
        print('数据加载成功')
        print('正常的session数：' + str(len(self.normal_blocks)))
        print('异常的session数：' + str(len(self.abnormal_blocks)))
        self.output_logkey_and_logvalue()
        print('数据生成成功')

hdfs_deeplog_preprocessor()
