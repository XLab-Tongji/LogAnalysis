import pandas as pd

logparser_structed_file = '../../Data/logparser_result/Drain/HDFS.log_structured.csv'
logparser_event_file = '../../Data/logparser_result/Drain/HDFS.log_templates.csv'
logparser_structed_file_loglizer = '../../Data/logparser_result/Drain/HDFS.log_structured(loglizer).csv'


def get_log_template_dic():
    dic = {}
    datafile = open(logparser_event_file, 'r', encoding='UTF-8')
    data = pd.read_csv(datafile)
    for _, row in data.iterrows():
        dic[row['EventId']] = row['numberID']
    return dic


logparser_result = pd.read_csv(logparser_structed_file, header=0)
dic = get_log_template_dic()
logparser_result['EventId'] = [dic.get(x) for x in logparser_result['EventId'].values]

logparser_result.to_csv(logparser_structed_file_loglizer, columns=logparser_result.columns, index=0, header=1)