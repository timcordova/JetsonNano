import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device('cuda:0')
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.i2o = nn.Linear(input_size + hidden_size, output_size).to(device)
        self.softmax = nn.LogSoftmax().to(device)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1).to(device)
        hidden = self.i2h(combined).to(device)
        output = self.i2o(combined).to(device)
        output = self.softmax(output).to(device)
        return output.to(device), hidden.to(device)

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size)).to(device)