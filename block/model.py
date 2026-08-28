import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import TransformerBlock


class BasicModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len)
            self.blocks.append(block)

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)

        x = self.token_embedding(x)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        logits = self.output(x)

        return logits


# Test
if __name__ == "__main__":
    vocab_size = 100
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 2
    max_seq_len = 16

    model = BasicModel(
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_seq_len
    )

    B = 2
    T = 8

    x = torch.randint(0, vocab_size, (B, T))
    y = torch.randint(0, vocab_size, (B, T))

    logits = model(x)

    print("input :", x.shape)
    print("output:", logits.shape)

    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    print("loss  :", loss.item())

    loss.backward()

    print("backward: OK")