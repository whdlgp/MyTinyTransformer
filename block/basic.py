import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedAttentionHead(nn.Module):
    def __init__(self, d_model, d_k, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # d_k: length of output per token
        # max_seq_len: Max length of input sequence

        super().__init__()
        self.d_k = d_k

        # W_q,k,v: (d_model, d_k)
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)

        # Not trainable. Mask for training
        # Shape: (max_seq_len, max_seq_len)
        # Example (4x4):
        # [1, 0, 0, 0]
        # [1, 1, 0, 0]
        # [1, 1, 1, 0]
        # [1, 1, 1, 1]
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

        # Dropout for training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # T: Sequence length (T <= max_seq_len)
        # Batch dimension (B) is omitted in comment for simplicity.
        # x: (T × d_model)
        B, T, _ = x.shape

        # (T × d_model) @ (d_model × d_k) = (T × d_k)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (T × d_k) @ (d_k × T) = (T × T)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        # Apply mask
        # Example (3x3):
        # Before:                  Mask:                  After:
        # [ 1.2,  0.5, -0.1]       [1, 0, 0]              [ 1.2, -inf, -inf]
        # [ 0.8,  2.1,  0.3]   |   [1, 1, 0]   ====>      [ 0.8,  2.1, -inf]
        # [-0.5,  0.2,  1.5]       [1, 1, 1]              [-0.5,  0.2,  1.5]
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # Final attention score
        attn = F.softmax(scores, dim=-1)

        # Dropout for training
        attn = self.dropout(attn)

        # (T × T) @ (T × d_k) = (T × d_k)
        out = attn @ V

        # (T × d_k)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # max_seq_len: Max length of input sequence

        super().__init__()

        # d_k: output length of each attention head
        # d_model == num_heads * d_k
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        d_k = d_model // num_heads

        # Multiple Attention Head
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head = MaskedAttentionHead(d_model, d_k, max_seq_len, dropout)
            self.heads.append(head)

        # W_o: (d_model, d_model), weights for mixing attn. outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # T: Sequence length (T <= max_seq_len)
        # Batch dimension (B) is omitted in comment for simplicity.
        # x: (T × d_model)

        # [(T × d_k), (T × d_k),,, (T × d_k)]
        head_outputs = []
        for head in self.heads:
            head_out = head(x)
            head_outputs.append(head_out)

        # [(T × d_k), (T × d_k),,, (T × d_k)] -> (T, d_model)
        concat = torch.cat(head_outputs, dim=-1)

        # (T, d_model) @ (d_model, d_model) = (T, d_model)
        out = self.W_o(concat)

        # (T, d_model)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        # d_model: length of embedding vector per token
        # d_ff: hidden layer size in feed-forward network
        super().__init__()

        # W_1: (d_model, d_ff)
        self.W_1 = nn.Linear(d_model, d_ff)
        # W_1: (d_ff, d_model)
        self.W_2 = nn.Linear(d_ff, d_model)

        # Dropout for training
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Batch dimension (B) is omitted in comment for simplicity.
        # x: (T × d_model)

        # (T × d_model) @ (d_model, d_ff) = (T x d_ff)
        x = self.W_1(x)
        # Activation
        x = F.relu(x)
        # Dropout for training
        x = self.dropout(x)
        # (T x d_ff) @ (d_ff, d_model) = (T × d_model)
        x = self.W_2(x)

        # (T × d_model)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # d_ff: hidden layer size in feed-forward network
        # max_seq_len: Max length of input sequence
        
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        #    x ──────┐
        #    │       │
        #    ▼       │
        # Attention  │
        #    │       │
        #    ▼       │
        #    ⊕ ◄────┘
        #    │
        #    ▼
        #   Norm
        #    │
        #    ├────────┐
        #    ▼        │
        # FeedForward │
        #    │        │
        #    ▼        │
        #    ⊕ ◄─────┘
        #    │
        #    ▼
        #   Norm
        #    │
        #    ▼
        #  output
    
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)

        return x