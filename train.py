import os
from tqdm import tqdm
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.optim import AdamW

from block.model import BasicModel
from tokenizer.simple_tokenizer import Tokenizer
from dataloader.tinyshakespeare import prepare_files, get_dataloader
from util.checkpoint import load_checkpoint, save_checkpoint
from util.config import load_config
from util.helper import get_device


class Trainer:
    def __init__(self, config):
        # Set Device
        self.m_cfg, self.t_cfg = config["model"], config["train"]
        self.device = get_device()

        # Download or load dataset
        train_path, val_path, test_path = prepare_files()

        # Read model/train config
        with open(train_path, "r", encoding="utf-8") as f:
            self.tokenizer = Tokenizer.from_text(f.read())

        # Dataloader
        self.train_loader = get_dataloader(train_path, self.tokenizer, self.m_cfg["max_seq_len"], self.t_cfg["batch_size"], shuffle=True)
        self.val_loader = get_dataloader(val_path, self.tokenizer, self.m_cfg["max_seq_len"], self.t_cfg["batch_size"], shuffle=False)

        # Model
        self.model = BasicModel(
                        self.tokenizer.vocab_size, self.m_cfg["d_model"], self.m_cfg["num_heads"],
                        self.m_cfg["d_ff"], self.m_cfg["num_layers"], self.m_cfg["max_seq_len"], self.m_cfg["dropout"]
                    ).to(self.device)
        
        # model 
        self.optimizer = AdamW(self.model.parameters(), lr=self.t_cfg["lr"], weight_decay=self.t_cfg["weight_decay"])

        # Load previous checkpoint
        self.checkpoint_path = self.t_cfg["checkpoint_path"]
        self.start_epoch = 0
        if os.path.exists(self.checkpoint_path):
            self.start_epoch = load_checkpoint(self.checkpoint_path, self.model, self.device, self.optimizer) + 1
            print(f"resumed from epoch {self.start_epoch}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        progress = tqdm(self.train_loader, desc="train")
        for inputs, labels in progress:
            # Inference
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = self.model(inputs)

            # Loss
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.tokenizer.vocab_size), labels.reshape(-1))

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # logging only, not used in training
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        for inputs, labels in self.val_loader:
            # Inference
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = self.model(inputs)

            # Loss
            loss = nn.functional.cross_entropy(logits.reshape(-1, self.tokenizer.vocab_size), labels.reshape(-1))
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self):
        start_train = time.time()

        for epoch in range(self.start_epoch, self.t_cfg["epochs"]):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            print(f"epoch {epoch} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

            save_checkpoint(self.checkpoint_path, self.model, self.optimizer, epoch)
            print(f"checkpoint saved: {self.checkpoint_path}")

        train_time = time.time() - start_train
        print(f"training finished in {timedelta(seconds=int(train_time))}")

def main():
    config = load_config()
    trainer = Trainer(config)
    trainer.fit()


if __name__ == "__main__":
    main()