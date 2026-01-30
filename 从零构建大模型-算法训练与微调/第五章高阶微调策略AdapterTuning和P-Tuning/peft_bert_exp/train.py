import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import BertTokenizer
from model.bert_wrapper import BertForPEFT
from config import ExpConfig
from eval import evaluate_with_metrics


# =========================
# 1. 构造一个可控的 Toy Dataset
# =========================

class ToyTextDataset(Dataset):
    """
    构造一个“有弱规律”的数据集：
    - 句子中 token id 偏大 → label = 1
    - token id 偏小 → label = 0
    这样模型是可以学到的
    """
    def __init__(self, tokenizer, num_samples=200, max_len=16):
        self.samples = []
        vocab_size = tokenizer.vocab_size

        for _ in range(num_samples):
            label = torch.randint(0, 2, (1,)).item()

            if label == 0:
                tokens = torch.randint(100, 1000, (max_len,))
            else:
                tokens = torch.randint(2000, 3000, (max_len,))

            attention_mask = torch.ones(max_len)
            self.samples.append((tokens, attention_mask, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, mask, label = self.samples[idx]
        return (
            input_ids.long(),
            mask.long(),
            torch.tensor(label).long()
        )


# =========================
# 2. 训练函数
# =========================

def train(
    model,
    train_loader,
    val_loader,
    config,
    device="cuda",
    epochs=5
):
    model.to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 验证
        metrics = evaluate_with_metrics(model, val_loader, device)

        print(
            f"[Epoch {epoch}] "
            f"Train loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"
        )


# =========================
# 3. 主入口（可直接运行）
# =========================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 切换这里即可对比四种方法
    config = ExpConfig(
        peft_type="lora",  # lora | adapter | prefix | ptuning
        lr=2e-4
    )

    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    train_set = ToyTextDataset(tokenizer, num_samples=500)
    val_set = ToyTextDataset(tokenizer, num_samples=200)

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32
    )

    model = BertForPEFT(config)

    print("Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n}: {p.numel()}")

    train(
        model,
        train_loader,
        val_loader,
        config,
        device=device,
        epochs=5
    )


if __name__ == "__main__":
    main()
