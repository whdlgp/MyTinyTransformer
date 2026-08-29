import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import FeedForward


class MultiHeadAttentionBatched(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # max_seq_len: Max length of input sequence

        super().__init__()

        # d_k: output length of each attention head
        # d_model == num_heads * d_k
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W_q,k,v: (d_model, d_model), all heads computed at once
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

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

        # W_o: (d_model, d_model), weights for mixing attn. outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # T: Sequence length (T <= max_seq_len)
        # Batch dimension (B) is omitted in comment for simplicity.
        # x: (T x d_model)
        B, T, _ = x.shape

        # (T x d_model) @ (d_model x d_model) = (T x d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (T x d_model) -> (T x num_heads x d_k) -> (num_heads x T x d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # (num_heads x T x d_k) @ (num_heads x d_k x T) = (num_heads x T x T)
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

        # (num_heads x T x T) @ (num_heads x T x d_k) = (num_heads x T x d_k)
        out = attn @ V

        # (num_heads x T x d_k) -> (T x num_heads x d_k) -> (T x d_model)
        # Note: B omitted. actual shape (num_heads x T x d_k): (B x num_heads x T x d_k).
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)

        # (T x d_model) @ (d_model x d_model) = (T x d_model)
        out = self.W_o(out)

        # (T x d_model)
        return out


class TransformerBlockBatched(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # d_ff: hidden layer size in feed-forward network
        # max_seq_len: Max length of input sequence

        super().__init__()

        self.attention = MultiHeadAttentionBatched(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)

        return x


class MultiHeadAttentionKVCache(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # max_seq_len: Max length of input sequence

        super().__init__()

        # d_k: output length of each attention head
        # d_model == num_heads * d_k
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W_q,k,v: (d_model, d_model), all heads computed at once
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

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

        # W_o: (d_model, d_model), weights for mixing attn. outputs
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past_kv=None):
        # T: Sequence length, Full or New token only. (T <= max_seq_len)
        # Batch dimension (B) is omitted in comment for simplicity.
        # x: (T x d_model)
        B, T, _ = x.shape

        # (T x d_model) @ (d_model x d_model) = (T x d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (T x d_model) -> (T x num_heads x d_k) -> (num_heads x T x d_k)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # past_kv + current kv. cache already reshaped, (num_heads x T x d_k)
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # Cache to return for the next step
        new_kv = (K, V)

        # T_full: past T + current T
        # (num_heads x T x d_k) @ (num_heads x d_k x T_full) = (num_heads x T x T_full)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        # Apply mask
        # Example (3x3):
        # Before:                  Mask:                  After:
        # [ 1.2,  0.5, -0.1]       [1, 0, 0]              [ 1.2, -inf, -inf]
        # [ 0.8,  2.1,  0.3]   |   [1, 1, 0]   ====>      [ 0.8,  2.1, -inf]
        # [-0.5,  0.2,  1.5]       [1, 1, 1]              [-0.5,  0.2,  1.5]
        #
        # T_full: past T + current T
        # offset: Mask offset. only current T
        T_full = K.shape[2]
        offset = T_full - T
        scores = scores.masked_fill(self.mask[offset:T_full, :T_full] == 0, float('-inf'))

        # Final attention score
        attn = F.softmax(scores, dim=-1)

        # Dropout for training
        attn = self.dropout(attn)

        # (num_heads x T x T_full) @ (num_heads x T_full x d_k) = (num_heads x T x d_k)
        out = attn @ V

        # (num_heads x T x d_k) -> (T x num_heads x d_k) -> (T x d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)

        # (T x d_model) @ (d_model x d_model) = (T x d_model)
        out = self.W_o(out)

        # (T x d_model), ((num_heads x T_full x d_k), (num_heads x T_full x d_k))
        return out, new_kv


class TransformerBlockKVCache(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, dropout):
        # d_model: length of embedding vector per token
        # num_heads: number of attention head
        # d_ff: hidden layer size in feed-forward network
        # max_seq_len: Max length of input sequence

        super().__init__()

        self.attention = MultiHeadAttentionKVCache(d_model, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attention(x, past_kv)
        x = x + attn_out
        x = self.norm1(x)

        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.norm2(x)

        return x, present_kv
