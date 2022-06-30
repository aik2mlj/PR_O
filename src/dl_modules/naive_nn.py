from torch import nn
# from torch.distributions import Normal


class NaiveNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NaiveNN, self).__init__()
        self.nn = nn.Linear(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        output = self.nn(x)
        return output
