from tcn import TemporalConvNet

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          nonlinearity='tanh', batch_first=True)
        # batch_first=True allow to give x such that its first dimension is the batch_size (as in CNN)

        self.fully_connected = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        # x dim : batch_size, sequence_lenght, input_size 

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # h0 dim : num_layers, batch_size, hidden_size

        out, _ = self.rnn(x.to(device), h0.to(device))  
        # out dim : batch_size, seq_lenght, hidden_size

        out = out[:, -1, :] 
        # out dim : batch_size, hidden_size
        
        out = self.fully_connected(out)
        # out dim : batch_size, num_classes

        return nn.functional.softmax(out, dim=1)
    

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fully_connected = nn.Linear(in_features=num_channels[-1], out_features=output_size)

    def forward(self, inputs):
        # x dim: batch_size, seq_lenght, num_channels[0]

        x = torch.transpose(inputs, 1, 2)
        # x dim: batch_size, num_channels[0], seq_lenght

        y = self.tcn(x)
        # y dim: batch_size, num_channels[0], seq_lenght

        out = y[:, :, -1]
        # out dim: batch_size, num_channels[-1]

        o = self.fully_connected(out)
        # o dim : batch_size, num_classes
        return nn.functional.log_softmax(o, dim=1)