import os
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.txt")
    val_path = os.path.join(DATA_DIR, "val.txt")
    test_path = os.path.join(DATA_DIR, "test.txt")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return train_path, val_path, test_path

    raw_path = os.path.join(DATA_DIR, "raw.txt")
    if not os.path.exists(raw_path):
        print("Getting the text file.")
        urllib.request.urlretrieve(SOURCE_URL, raw_path)

    with open(raw_path, "r", encoding="utf-8") as f:
        text = f.read()

    n_train = int(0.8 * len(text))
    n_val = int(0.9 * len(text))

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(text[:n_train])
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(text[n_train:n_val])
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(text[n_val:])

    return train_path, val_path, test_path


class CharDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        input_ids = self.data[idx : idx + self.seq_len]
        labels = self.data[idx + 1 : idx + self.seq_len + 1]
        return input_ids, labels


def get_dataloader(path, tokenizer, seq_len, batch_size, shuffle):
    dataset = CharDataset(path, tokenizer, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
