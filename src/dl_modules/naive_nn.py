from torch import nn
import torch.distributions
from torch.distributions import Normal


class NaiveNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NaiveNN, self).__init__()
        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_var = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, dist: Normal):
        mu = dist.mean
        var = dist.variance
        output_mu = self.linear_mu(mu)
        output_var = self.linear_var(var).exp_()
        # print(output_mu.size(), output_var.size())
        return Normal(output_mu, output_var)
