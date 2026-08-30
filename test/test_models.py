import torch
import torch.nn.functional as F

from block.model import MODEL_REGISTRY


if __name__ == "__main__":
    vocab_size = 100
    d_model = 32
    num_heads = 4
    d_ff = 128
    num_layers = 2
    max_seq_len = 16
    dropout = 0.1

    B = 2
    T = 8

    x = torch.randint(0, vocab_size, (B, T))
    y = torch.randint(0, vocab_size, (B, T))

    for name, model_cls in MODEL_REGISTRY.items():
        print(f"===== {name} =====")

        model = model_cls(
            vocab_size,
            d_model,
            num_heads,
            d_ff,
            num_layers,
            max_seq_len,
            dropout
        )

        out = model(x)
        logits = out.logits

        print("input :", x.shape)
        print("output:", logits.shape)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        print("loss  :", loss.item())

        loss.backward()

        print("backward: OK")
        print()