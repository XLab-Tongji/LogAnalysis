import configparser
import json
import logging
import sys
import os
import shutil

from logparsing.drain.drain3.template_miner import TemplateMiner
from logparsing.drain.drain3.file_persistence import FilePersistence


def get_hdfs_drain_clusters(log,drain_out,bin_dir):
    persistence_type = "FILE"
    config = configparser.ConfigParser()
    config.read('drain3.ini')
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    persistence = FilePersistence(bin_dir)
    template_miner = TemplateMiner(persistence)
    shutil.rmtree(drain_out)
    os.makedirs(drain_out,exist_ok=True)
    with open(log,'r') as file:
        lineNum = 0
        for line in file.readlines():
            print(lineNum)
            result = template_miner.add_log_message(line)
            cluster_id = json.dumps(result["cluster_id"])
            cluster_id = int(cluster_id[2:-1])
            with open(drain_out+str(cluster_id),'a') as outfile:
                outfile.write(str(lineNum) + " ")
            lineNum += 1
    # print("Clusters:")
    #for cluster in template_miner.drain.clusters:
        #print(cluster)
