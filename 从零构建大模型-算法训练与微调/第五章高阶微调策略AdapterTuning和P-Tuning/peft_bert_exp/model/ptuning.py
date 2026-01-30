import torch
import torch.nn as nn

class PromptEmbedding(nn.Module):
    def __init__(self, prompt_len, hidden):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_len, hidden))

    def forward(self, batch_size):
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
