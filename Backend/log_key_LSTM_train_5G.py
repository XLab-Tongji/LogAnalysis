import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# use cuda if available  otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train parameters
window_length = 4
input_size = 1
hidden_size = 20
num_of_layers = 3
root_path = "../5G/LogClusterResult-5G/"

model_output_directory = root_path + 'output/model1'
log_directory = root_path + 'output/log1'

num_epochs = 1000  # 300
batch_size = 200  # 2048
log_template = 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs)
train_file_name = 'logkey/logkey_train'
data_file = root_path + train_file_name


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


def generate_seq_label(file_path):
    num_of_sessions = 0
    input_data, output_data = [], []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            num_of_sessions += 1
            line = tuple(map(lambda n: n, map(int, line.strip().split())))
            for i in range(len(line) - window_length):
                input_data.append(line[i:i + window_length])
                output_data.append(line[i + window_length])
    data_set = TensorDataset(torch.tensor(input_data, dtype=torch.float), torch.tensor(output_data))
    return data_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_of_layers', default=num_of_layers, type=int)
    parser.add_argument('-hidden_size', default=hidden_size, type=int)
    parser.add_argument('-window_length', default=window_length, type=int)

    args = parser.parse_args()

    num_of_layers = args.num_of_layers
    hidden_size = args.hidden_size
    window_length = args.window_length

    num_classes = len(os.listdir(root_path + 'clusters/')) + 2
    print("Train num_classes: ", num_classes)
    model = Model(input_size, hidden_size, num_of_layers, num_classes).to(device)
    sequence_data_set = generate_seq_label(data_file)
    data_loader = DataLoader(sequence_data_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    writer = SummaryWriter(logdir = log_directory + '/' + log_template)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    total_step = len(data_loader)
    outfile = './output.txt'
    for epoch in range(num_epochs):
        with open(outfile, 'w') as file:
            file.write('4 '+str(epoch / (num_epochs-1) *100))
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
