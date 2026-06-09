import torch
from torch import nn
import torch.nn.functional as F
import math

# NET ------
#noisy dqn

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features  = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.weight_mu = nn.Parameter(
                torch.empty(
                    out_features, 
                    in_features, 
                    device=device, 
                    dtype=torch.float32, 
                    requires_grad=True
                    )
                )
        self.weight_sigma = nn.Parameter(
                torch.empty(
                    out_features, 
                    in_features, 
                    device=device, 
                    dtype=torch.float32,
                    requires_grad=True
                    )
                )
        self.register_buffer('weight_epsilon', 
                torch.empty(
                    out_features, 
                    in_features, 
                    device=device, 
                    dtype=torch.float32, 
                    )
                )

        self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features, 
                    device=device, 
                    dtype=torch.float32, 
                    requires_grad=True
                    )
                )
        self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features, 
                    device=device, 
                    dtype=torch.float32,
                    requires_grad=True
                    )
                )
        self.register_buffer('bias_epsilon', 
                torch.empty(
                    out_features, 
                    device=device, 
                    dtype=torch.float32
                    )
                )

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias   = self.bias_mu   + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(torch.nn.Module):
    def __init__(self):
        super(NoisyDQN, self).__init__()
        self.c1 = torch.nn.Conv2d(in_channels=4, out_channels = 32, kernel_size=8,stride=4)
        self.c2 = torch.nn.Conv2d(in_channels=32, out_channels = 64 ,kernel_size=4,stride=2)
        self.c3 = torch.nn.Conv2d(in_channels=64, out_channels = 64 ,kernel_size=3,stride=1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = NoisyLinear(in_features=3136, out_features=512)
        self.fc2 = NoisyLinear(in_features=512, out_features=12)
        self.relu = torch.nn.ReLU()

    def reset_noise(self):
        self.fc1.reset_noise()
        self.fc2.reset_noise()

    def forward(self, x):
        out_conv = self.c1(x)
        out_conv = self.relu(out_conv)
        out_conv = self.c2(out_conv)
        out_conv = self.relu(out_conv)
        out_conv = self.c3(out_conv)
        out_conv = self.relu(out_conv)
        flat = self.flatten(out_conv)
        out = self.fc1(flat)
        out = self.relu(out)
        out = self.fc2(out)

        return out
