import torch

import torch.nn as nn

class UniRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(UniRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage:
# model = UniRNN(input_size=10, hidden_size=20, output_size=1, num_layers=2)
# x = torch.randn(5, 3, 10)  # (batch_size, sequence_length, input_size)
# output = model(x)
# print(output)