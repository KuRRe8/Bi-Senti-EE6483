import torch

import torch.nn as nn

class RNNModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModule, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Example usage:
# model = RNNModule(input_size=10, hidden_size=20, output_size=1)
# input_tensor = torch.randn(5, 3, 10)  # (batch_size, sequence_length, input_size)
# output = model(input_tensor)
# print(output)