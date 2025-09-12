from typing import Dict, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    preds = []
    probs = []
    gts = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            p = torch.softmax(logits, dim=1)[:, 1]
            preds.append((p > 0.5).long().cpu())
            probs.append(p.cpu())
            gts.append(labels.cpu())
    preds = torch.cat(preds)
    probs = torch.cat(probs)
    gts = torch.cat(gts)

    try:
        auc = roc_auc_score(gts.numpy(), probs.numpy())
    except Exception:
        auc = float('nan')

    acc = accuracy_score(gts.numpy(), preds.numpy())
    avg_loss = total_loss / len(loader.dataset)
    return {"val_loss": avg_loss, "val_acc": acc, "val_auc": auc}
