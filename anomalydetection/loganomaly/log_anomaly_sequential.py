import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
from . import *

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train parameters
window_length = 4
input_size = 1
hidden_size = 20
num_of_layers = 2
root_path = "../Data/LogClusterResult-k8s/"

model_output_directory = root_path + 'output/model1'
log_directory = root_path + 'output/log1'

num_epochs = 1000  # 300
batch_size = 200  # 2048
log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
train_file_name = 'logkey/logkey_train'
data_file = root_path + train_file_name

