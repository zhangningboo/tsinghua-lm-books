import torch
import torch.nn as nn

class PrefixEncoder(nn.Module):
    def __init__(self, prefix_len, hidden):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(prefix_len, hidden))

    def forward(self, batch_size):
        return self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
