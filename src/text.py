import re
from collections import Counter

import numpy as np

AFRIBERTA = "castorini/afriberta_base"


def word_tokenize(text):
    return re.findall(r"[a-zà-ÿ0-9']+", str(text).lower())


class WordVocab:
    PAD, UNK = 0, 1

    def __init__(self, texts, min_freq=2, max_size=40000):
        counts = Counter(tok for t in texts for tok in word_tokenize(t))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.PAD

    def __len__(self):
        return len(self.itos)

    def encode(self, texts, max_len=512):
        return [[self.stoi.get(tok, self.UNK) for tok in word_tokenize(t)][:max_len] or [self.UNK]
                for t in texts]

    def oov_rate(self, texts):
        toks = [tok for t in texts for tok in word_tokenize(t)]
        return sum(tok not in self.stoi for tok in toks) / max(len(toks), 1)


def load_subword_tokenizer(name=AFRIBERTA):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 512
    return tokenizer


def subword_encode(texts, tokenizer, max_len=512):
    enc = tokenizer(list(texts), add_special_tokens=False, truncation=True, max_length=max_len)
    return enc["input_ids"]


def subword_lengths(texts, tokenizer):
    enc = tokenizer(list(texts), add_special_tokens=False, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]])


def transformer_encode(texts, tokenizer, max_len=512, strategy="head", head_len=128):
    body = max_len - 2
    ids = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    out = []
    for seq in ids:
        if len(seq) > body:
            seq = seq[:body] if strategy == "head" else seq[:head_len] + seq[-(body - head_len):]
        out.append([tokenizer.cls_token_id] + seq + [tokenizer.sep_token_id])
    return out


def truncate_to_subwords(texts, tokenizer, max_len=510):
    ids = subword_encode(texts, tokenizer, max_len)
    return [tokenizer.decode(seq) for seq in ids]
