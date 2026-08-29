import torch
import torch.nn as nn

from .basic import TransformerBlock
from .optimized import TransformerBlockBatched


class BasicModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            self.blocks.append(block)

        self.output = nn.Linear(d_model, vocab_size)

        # Dropout for training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)

        x = self.token_embedding(x)
        x = x + self.position_embedding(positions)

        # Dropout for training
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.output(x)

        return logits


class BatchedHeadModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlockBatched(d_model, num_heads, d_ff, max_seq_len, dropout)
            self.blocks.append(block)

        self.output = nn.Linear(d_model, vocab_size)

        # Dropout for training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)

        x = self.token_embedding(x)
        x = x + self.position_embedding(positions)

        # Dropout for training
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.output(x)

        return logits


MODEL_REGISTRY = {
    "basic": BasicModel,
    "batched": BatchedHeadModel,
}
