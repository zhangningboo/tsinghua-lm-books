

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# Adapter模块的定义
class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        """
        Arg:
            input_dim: 输入特征的维度
            bottleneck_dim: 瓶颈层的维度，通常远小于input_dim
        """
        super(Adapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        down = self.down_proj(x)
        activated = self.activation(down)
        up = self.up_proj(activated)
        return x + up  # 残差连接，使模型更稳定


class BertLayerWithAdapter(nn.Module):
    def __init__(self, bert_layer, adapter):
        super().__init__()
        self.layer = bert_layer
        self.adapter = adapter

    def forward(self, hidden_states, *args, **kwargs):
        # 调用原始 BertLayer
        output = self.layer(hidden_states, *args, **kwargs)

        # BertLayer 返回的是 hidden_states（不是 tuple）
        hidden_states = output

        # Adapter
        hidden_states = self.adapter(hidden_states)

        return hidden_states


# 主模型：正确集成 Adapter
class BertWithAdapter(nn.Module):
    def __init__(self, adapter_dim=64, num_labels=2, pretrained_model_name='../bert-base-uncased'):
        super().__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)

        # 冻结 BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        # Adapter
        self.adapter_layers = nn.ModuleList()
        for i in range(self.bert.config.num_hidden_layers):
            adapter = Adapter(
                input_dim=self.bert.config.hidden_size,
                bottleneck_dim=adapter_dim
            )
            self.adapter_layers.append(adapter)
            self.bert.encoder.layer[i] = BertLayerWithAdapter(
                self.bert.encoder.layer[i],
                adapter
            )

        # 分类头（可训练）
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # [CLS] 向量
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_embedding)
        return logits


tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased')

# 模拟数据加载
def prepare_data(sentences, labels, tokenizer, max_length=128):
    encoding = tokenizer(
        sentences,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt',
        attention_mask=True,
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_mask, labels)

# 准备示例数据
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]
dataset = prepare_data(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct_predictions = 0, 0
    for batch in dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        outputs = model(input_ids, attention_mask)
        logits = outputs
        loss = criterion(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = torch.max(logits, dim=1)
        correct_predictions += torch.sum(pred == labels).item()
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy


truth_model = BertWithAdapter(adapter_dim=64)

for name, param in truth_model.named_parameters():
    if param.requires_grad:
        print(f"✅ 可训练: {name} (shape={param.shape})")
    else:
        # 检查是否属于 BERT 主干
        if 'bert' in name and 'adapter' not in name.lower():
            assert not param.requires_grad, f"❌ BERT 参数意外可训练: {name}"

# %%
# 训练过程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
truth_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_truth = optim.Adam(
    list(truth_model.adapter_layers.parameters()) +
    list(truth_model.classifier.parameters()),
    lr=1e-4
)

for epoch in range(300):  # 训练3个epoch
    loss, acc = train_epoch(truth_model, dataloader, criterion, optimizer_truth, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
