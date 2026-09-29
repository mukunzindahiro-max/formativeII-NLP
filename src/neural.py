import contextlib
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from .data import LABELS, SEED, set_seed
from .evaluate import macro_f1
from .text import AFRIBERTA

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def encode_labels(categories):
    return np.array([LABEL_TO_ID[c] for c in categories])


def class_weights(categories, power=0.5):
    y = encode_labels(categories)
    weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y) ** power
    return torch.tensor(weights / weights.mean(), dtype=torch.float)


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i], self.labels[i]


def make_loader(sequences, categories, pad_id, batch_size=32, shuffle=False):
    labels = encode_labels(categories)

    def collate(batch):
        seqs, ys = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs])
        ids = torch.full((len(seqs), max(int(lengths.max()), 5)), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        mask = torch.arange(ids.size(1))[None, :] < lengths[:, None]
        return ids, mask.long(), torch.tensor(ys)

    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(SequenceDataset(sequences, labels), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate, generator=generator)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, pad_id, n_classes=len(LABELS), emb_dim=128,
                 kernel_sizes=(3, 4, 5), n_filters=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, ids, mask):
        x = self.embedding(ids).transpose(1, 2)
        lengths = mask.sum(dim=1)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            k = conv.kernel_size[0]
            valid = torch.arange(h.size(2), device=h.device)[None, :] <= (lengths[:, None] - k)
            valid[:, 0] = True
            h = h.masked_fill(~valid[:, None, :], -1e4)
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, pad_id, n_classes=len(LABELS), emb_dim=128,
                 hidden=128, pooling="last", dropout=0.3):
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, ids, mask):
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(self.embedding(ids), lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, _) = self.lstm(packed)
        if self.pooling == "last":
            rep = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            scores = self.attention(out).squeeze(-1).masked_fill(mask == 0, -1e4)
            weights = torch.softmax(scores, dim=1)
            rep = (weights.unsqueeze(-1) * out).sum(dim=1)
        return self.fc(self.dropout(rep))


class TransformerClassifier(nn.Module):
    def __init__(self, name=AFRIBERTA, n_classes=len(LABELS)):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=n_classes)

    def forward(self, ids, mask):
        return self.model(input_ids=ids, attention_mask=mask).logits


def predict_proba(model, loader, device):
    model.eval()
    probs, losses = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            with autocast(device):
                logits = model(ids.to(device), mask.to(device))
            logits = logits.float()
            losses.append(F.cross_entropy(logits, y.to(device), reduction="sum").item())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs), sum(losses) / len(loader.dataset)


def train_model(model, train_loader, dev_loader, epochs, optimizer, scheduler=None,
                weights=None, patience=3, grad_clip=1.0, device=None):
    device = device or get_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    dev_true = np.array(LABELS)[[y for _, y in dev_loader.dataset]]
    best_score, best_state, history, bad_epochs = -1.0, None, [], 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in train_loader:
            optimizer.zero_grad()
            with autocast(device):
                loss = loss_fn(model(ids.to(device), mask.to(device)).float(), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * len(y)
        dev_proba, dev_loss = predict_proba(model, dev_loader, device)
        dev_f1 = macro_f1(dev_true, np.array(LABELS)[dev_proba.argmax(axis=1)])
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset),
                        "dev_loss": dev_loss, "dev_macro_f1": dev_f1})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.3f}, "
              f"dev loss {dev_loss:.3f}, dev macro-F1 {dev_f1:.3f}")
        if dev_f1 > best_score:
            best_score, best_state, bad_epochs = dev_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping after epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return {"model": model, "history": history, "dev_macro_f1": best_score,
            "train_time_s": time.time() - start}


def run_experiment(build_model, train_seqs, dev_seqs, val_seqs, fit_df, dev_df, val_df, pad_id,
                   epochs=15, lr=1e-3, batch_size=32, weights=None, patience=3,
                   weight_decay=0.0, warmup=0.0):
    set_seed()
    device = get_device()
    train_loader = make_loader(train_seqs, fit_df["category"], pad_id, batch_size, shuffle=True)
    dev_loader = make_loader(dev_seqs, dev_df["category"], pad_id, batch_size)
    val_loader = make_loader(val_seqs, val_df["category"], pad_id, batch_size)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if warmup > 0:
        from transformers import get_linear_schedule_with_warmup
        steps = epochs * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * steps), steps)
    run = train_model(model, train_loader, dev_loader, epochs, optimizer, scheduler,
                      weights, patience, device=device)
    run["val_proba"], _ = predict_proba(run["model"], val_loader, device)
    run["n_params"] = sum(p.numel() for p in model.parameters())
    return run
