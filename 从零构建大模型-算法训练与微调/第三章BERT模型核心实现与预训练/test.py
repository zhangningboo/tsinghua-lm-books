import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 自注意力机制实现
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size必须是heads的整数倍"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        return self.fc_out(out)

# 掩码任务数据集创建函数
def create_masked_data(inputs, mask_token_id, vocab_size, mask_prob=0.15):
    inputs_with_masks = inputs.clone()
    labels = inputs.clone()
    
    for i in range(inputs.size(0)):
        for j in range(inputs.size(1)):
            prob = random.random()
            if prob < mask_prob:
                prob /= mask_prob
                if prob < 0.8:
                    inputs_with_masks[i, j] = mask_token_id  # 80%的概率替换为[MASK]
                elif prob < 0.9:
                    inputs_with_masks[i, j] = random.randint(0, vocab_size - 1)  # 10%的概率替换为随机词
                # 10%的概率保持不变
            else:
                labels[i, j] = -100  # 忽略不被遮掩的位置

    return inputs_with_masks, labels

# BERT编码器模块
class BERTEncoder(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, num_layers, dropout):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, forward_expansion, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Transformer编码器块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# 模拟数据
vocab_size = 30522  # BERT的词汇表大小
embed_size = 768
num_layers = 12
heads = 12
forward_expansion = 4
dropout = 0.1
seq_length = 20
batch_size = 2
mask_token_id = 103  # [MASK]标记的ID

# 初始化BERT编码器
bert_encoder = BERTEncoder(embed_size, heads, forward_expansion, num_layers, dropout)

# 随机生成输入数据
input_data = torch.randint(0, vocab_size, (batch_size, seq_length))
mask = torch.ones(batch_size, seq_length, seq_length)  # 无掩码

# 创建掩码任务数据
masked_data, labels = create_masked_data(input_data, mask_token_id, vocab_size)

# 前向传播
output = bert_encoder(masked_data, mask)
print("编码器输出形状:", output.shape)
print("掩码后的输入:\n", masked_data)
print("掩码标签:\n", labels)
print("编码器输出:\n", output)