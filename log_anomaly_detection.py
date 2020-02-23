from extractfeature.k8s import log_preprocessor
from extractfeature.k8s import  value_extract
import os
from logparsing.fttree import  fttree

log_file_dir = './Data/log/hdfs/'
log_file_name = 'HDFS_split'
log_fttree_out_directory = './Data/FTTreeResult-HDFS/clusters/'

if not os.path.exists(log_fttree_out_directory):
    os.makedirs(log_fttree_out_directory)

fttree.pattern_extract(log_file_dir, log_file_name, log_fttree_out_directory, 5, 4, 2)

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

