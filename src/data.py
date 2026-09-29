import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SPLIT_PATH = DATA_DIR / "split_ids.csv"

LABELS = ["Kitaifa", "michezo", "Biashara", "Kimataifa", "Burudani"]
MAJOR_LABELS = ["Kitaifa", "michezo", "Biashara"]
SEED = 42

for folder in (RESULTS_DIR, FIGURES_DIR):
    folder.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_raw(name="Train.csv"):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Train.csv and Test.csv from "
            f"https://zindi.world/competitions/swahili-news-classification-challenge/data "
            f"and put them in {DATA_DIR}."
        )
    return pd.read_csv(path)


def clean_content(text):
    text = str(text).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parts = ast.literal_eval(text)
            text = " ".join(str(p).strip() for p in parts)
        except (ValueError, SyntaxError):
            text = text.strip("[]").replace("', '", " ").strip("'")
    return " ".join(text.split())


def load_clean_train(min_words=3):
    df = load_raw("Train.csv")
    df["content"] = df["content"].apply(clean_content)
    keep = df["content"].str.split().apply(len) >= min_words
    return df[keep].reset_index(drop=True)


def make_split(df, val_size=0.2, seed=SEED):
    train_ids, val_ids = train_test_split(
        df["id"], test_size=val_size, stratify=df["category"], random_state=seed
    )
    split = pd.DataFrame({
        "id": pd.concat([train_ids, val_ids]),
        "split": ["train"] * len(train_ids) + ["val"] * len(val_ids),
    })
    split.to_csv(SPLIT_PATH, index=False)
    return split


def load_split():
    df = load_clean_train()
    split = pd.read_csv(SPLIT_PATH) if SPLIT_PATH.exists() else make_split(df)
    df = df.merge(split, on="id")
    train = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    val = df[df["split"] == "val"].drop(columns="split").reset_index(drop=True)
    return train, val


def dev_split(train_df, dev_size=0.1, seed=SEED):
    fit_df, dev_df = train_test_split(
        train_df, test_size=dev_size, stratify=train_df["category"], random_state=seed
    )
    return fit_df.reset_index(drop=True), dev_df.reset_index(drop=True)


def word_count(texts):
    return pd.Series(texts).str.split().apply(len).to_numpy()
