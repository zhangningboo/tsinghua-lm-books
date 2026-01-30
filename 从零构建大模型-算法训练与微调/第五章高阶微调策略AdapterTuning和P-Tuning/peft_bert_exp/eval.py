import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score


@torch.no_grad()
def evaluate_with_metrics(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": (all_preds == all_labels).mean(),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
    }
