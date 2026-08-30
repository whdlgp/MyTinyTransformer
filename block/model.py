import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any, Optional

from .basic import TransformerBlock
from .optimized import TransformerBlockBatched, TransformerBlockKVCache


@dataclass
class ModelOutput:
    logits: torch.Tensor
    cache: Optional[Any] = None


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

        return ModelOutput(logits=logits)


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

        return ModelOutput(logits=logits)


class KVCacheModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlockKVCache(d_model, num_heads, d_ff, max_seq_len, dropout)
            self.blocks.append(block)

        self.output = nn.Linear(d_model, vocab_size)

        # Dropout for training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None):
        B, T = x.shape

        # past_kv: [(K_0, V_0), (K_1, V_1), ...]
        # past_kv[0][0].shape: First K's shape, (B x num_heads x T x d_k)
        # offset: length of past T
        offset = past_kv[0][0].shape[2] if past_kv is not None else 0
        positions = torch.arange(offset, offset + T, device=x.device)

        x = self.token_embedding(x)
        x = x + self.position_embedding(positions)

        # Dropout for training
        x = self.dropout(x)

        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        new_kv = []
        for block, block_past_kv in zip(self.blocks, past_kv):
            x, block_new_kv = block(x, block_past_kv)
            new_kv.append(block_new_kv)

        logits = self.output(x)

        return ModelOutput(logits=logits, cache=new_kv)


MODEL_REGISTRY = {
    "basic": BasicModel,
    "batched": BatchedHeadModel,
    "kvcached": KVCacheModel,
}
