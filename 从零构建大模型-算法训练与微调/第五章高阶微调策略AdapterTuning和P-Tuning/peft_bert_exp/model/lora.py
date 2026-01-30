import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=32):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False

        self.A = nn.Linear(base_linear.in_features, r, bias=False)
        self.B = nn.Linear(r, base_linear.out_features, bias=False)
        self.scaling = alpha / r

        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.B(self.A(x))
