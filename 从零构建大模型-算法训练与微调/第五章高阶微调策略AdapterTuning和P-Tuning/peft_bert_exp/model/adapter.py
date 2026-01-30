import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, hidden, r):
        super().__init__()
        self.down = nn.Linear(hidden, r)
        self.up = nn.Linear(r, hidden)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
