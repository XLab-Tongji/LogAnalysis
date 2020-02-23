import os
# 把异常的删除就是正常的 暂时
# 选取没有异常的块
# cluster里读出来有正常的 也有异常的 根据anomaly csv 区别

def preprocessor_hdfs_ft(cluster_directory, anomaly_file_path):
    anomaly_log_lines = set()
    with open(anomaly_file_path, 'r') as anomaly_file:
        line = anomaly_file.readline()
        lines_str = line.split(' ')
        anomaly_log_lines.add([int(x) for x in lines_str])
    print(anomaly_log_lines)

    log_cluster = []
    file_names = os.listdir(cluster_directory)
    for file_name in file_names:
        with open(cluster_directory + file_name, 'r') as cluster:
            lines = cluster.readlines()
            line_numbers = [int(x) for x in lines[2].split(' ')]
            for number in line_numbers:
                log_cluster[number] = lines[0]

    with open(log_fttree_out_file, 'w') as file_obj:
    for i in sorted(log_cluster):
        file_obj.write(str(log_cluster[i]))
        if i % 10 == 0:
            file_obj.write('\n')
        else:
            file_obj.write(' ')