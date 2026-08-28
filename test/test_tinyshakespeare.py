import torch

from dataloader.tinyshakespeare import prepare_files, CharDataset, get_dataloader
from tokenizer.simple_tokenizer import Tokenizer


if __name__ == "__main__":
    train_path, val_path, test_path = prepare_files()

    print("train:", train_path)
    print("val  :", val_path)
    print("test :", test_path)
    print("prepare_files: OK")

    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer.from_text(text)

    seq_len = 32
    batch_size = 4

    dataset = CharDataset(train_path, tokenizer, seq_len)

    print("dataset size:", len(dataset))
    print("vocab size  :", tokenizer.vocab_size)

    input_ids, labels = dataset[0]

    print("input shape :", input_ids.shape)
    print("label shape :", labels.shape)

    assert input_ids.shape == (seq_len,)
    assert labels.shape == (seq_len,)
    assert torch.equal(labels[:-1], input_ids[1:])

    print("CharDataset: OK")

    dataloader = get_dataloader(train_path, tokenizer, seq_len, batch_size, shuffle=False)

    batch_inputs, batch_labels = next(iter(dataloader))

    print("batch input :", batch_inputs.shape)
    print("batch label :", batch_labels.shape)

    assert batch_inputs.shape == (batch_size, seq_len)
    assert batch_labels.shape == (batch_size, seq_len)
    assert torch.equal(batch_labels[:, :-1], batch_inputs[:, 1:])

    print("DataLoader: OK")