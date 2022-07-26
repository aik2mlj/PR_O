from torch import nn
import torch.distributions
from torch.distributions import Normal


class NaiveNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NaiveNN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, z_x):
        output = self.linear(z_x)
        # print(output_mu.size(), output_var.size())
        return output
