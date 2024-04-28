import torch as t
import torch.nn as nn

class PlanningNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlanningNetwork, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, obs, hidden):
        _input = t.concatenate((x, obs), dim=1)
        hidden = nn.functional.tanh(self.i2h(_input) + self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden