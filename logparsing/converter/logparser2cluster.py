# coding:utf-8
import pandas as pd
import os

# log parser_file should be structed.csv output should be './Data/FTTreeResult-HDFS/clusters/'
def logparser2cluster(logparser_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logparser_result = pd.read_csv(logparser_file, header=0)
    key_dict = {}
    value_dict = {}
    for _, row in logparser_result.iterrows():
        key = row['EventTemplate']
        if not key in key_dict:
            key_dict[key] = []
        key_dict[key].append(str(row['LineId']))
    key_num = 1
    for key, lines in key_dict.items():
        with open(output_dir + "/" + str(key_num), 'w') as f:
            f.write(key + "\n")
            f.write(" ".join(lines))
        key_num += 1

if __name__ == "__main__":
    logparser2cluster("Drain_result/HDFS.log_structured.csv", "clusters")
