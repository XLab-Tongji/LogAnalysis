import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import os

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_of_layers, num_of_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_of_layers = num_of_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_of_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_of_keys)

    def forward(self, input):
        # initial value of h and c
        h0 = torch.zeros(self.num_of_layers, input.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_of_layers, input.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(input, (h0, c0))
        # the output of final time step
        out = self.fc(out[:, -1, :])
        return out


def generate_seq_label(file_path,window_length):
    num_of_sessions = 0
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            num_of_sessions += 1
            line = list(map(lambda n: n, map(int, line.strip().split())))
            line = line + [-1] * (window_length + 1 - len(line))
            for i in range(len(line) - window_length):
                input_data.append(line[i:i + window_length])
                output_data.append(line[i + window_length])
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set

    

def train_model1(model_dir,log_preprocessor_dir,log_fttree_out_dir,num_epochs,batch_size,window_length,input_size,hidden_size,num_of_layers):
    model_output_directory = model_dir + 'model1'
    log_directory = model_dir + 'log1'
    log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
    train_file_name = log_preprocessor_dir+'logkey/logkey_train'
    data_file = train_file_name
    # 加 1 是因为 key 是从 1 开始算的
    num_classes = len(os.listdir(log_fttree_out_dir)) + 1
    print("Train num_classes: ", num_classes)
    model = Model(input_size, hidden_size, num_of_layers, num_classes).to(device)
    
    sequence_data_set = generate_seq_label(data_file,window_length)
    
    data_loader = DataLoader(sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    writer = SummaryWriter(logdir = log_directory + '/' + log_template)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        train_loss = 0
        for step, (seq, label) in enumerate(data_loader):
            # Forward
            seq = seq.clone().detach().view(-1, window_length, input_size).to(device)
            output = model(seq)
            # print("output:")
            # print(output)
            # print("label:")
            # print(label)

            # calculate loss
            loss = criterion(output, label.to(device))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('Epoch [{}/{}], training_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / len(data_loader.dataset)))
        writer.add_scalar('train_loss', train_loss / len(data_loader.dataset), epoch + 1)
        # save every 100 epoch
        if (epoch + 1) % 100 == 0:
            if not os.path.isdir(model_output_directory):
                os.makedirs(model_output_directory)
            e_log = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(epoch+1)
            torch.save(model.state_dict(), model_output_directory + '/' + e_log + '.pt')
    writer.close()
    print('Training finished')


