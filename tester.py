import os
import torch
import torch.nn.functional as F

from block.model import MODEL_REGISTRY
from tokenizer.simple_tokenizer import Tokenizer
from dataloader.tinyshakespeare import prepare_files, get_dataloader
from util.checkpoint import load_checkpoint
from util.config import load_config
from util.helper import get_device


class Tester:
    def __init__(self, config):
        # Set Device
        self.m_cfg, self.t_cfg, self.test_cfg = config["model"], config["train"], config["test"]
        self.device = get_device()

        # Download or load dataset
        train_path, val_path, test_path = prepare_files()
        self.test_path = test_path

        # Tokenizer (rebuilt from train.txt, same as training)
        with open(train_path, "r", encoding="utf-8") as f:
            self.tokenizer = Tokenizer.from_text(f.read())

        # Model
        model_cls = MODEL_REGISTRY[self.m_cfg["type"]]
        self.model = model_cls(
                        self.tokenizer.vocab_size, self.m_cfg["d_model"], self.m_cfg["num_heads"],
                        self.m_cfg["d_ff"], self.m_cfg["num_layers"], self.m_cfg["max_seq_len"], self.m_cfg["dropout"]
                    ).to(self.device)

        # Load trained checkpoint
        checkpoint_path = self.test_cfg["checkpoint_path"]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self):
        test_loader = get_dataloader(self.test_path, self.tokenizer, self.m_cfg["max_seq_len"], self.t_cfg["batch_size"], shuffle=False)
        total_loss = 0.0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            logits = self.model(inputs).logits
            loss = F.cross_entropy(logits.reshape(-1, self.tokenizer.vocab_size), labels.reshape(-1))
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        print(f"test loss: {avg_loss:.4f} | perplexity: {perplexity:.2f}")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=200, temperature=0.8):
        max_seq_len = self.m_cfg["max_seq_len"]
        ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)

        model_input = ids[:, -max_seq_len:]
        cache = None

        for _ in range(max_new_tokens):
            if ids.shape[1] >= max_seq_len:
                break

            if cache is not None:
                out = self.model(model_input, past_kv=cache)
            else:
                out = self.model(model_input)

            probs = F.softmax(out.logits[:, -1, :] / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)

            cache = out.cache

            if cache is not None:
                model_input = next_id
            else:
                model_input = ids[:, -max_seq_len:]

        return self.tokenizer.decode(ids[0].tolist())

    def chat(self):
        print("chat mode, type a prompt (or 'exit' to quit)")

        while True:
            prompt = input("> ")
            if prompt.strip().lower() == "exit":
                break
            
            # cache resets every turn
            output = self.generate(prompt)
            print(output)


def main():
    config = load_config()
    tester = Tester(config)

    tester.evaluate()
    tester.chat()


if __name__ == "__main__":
    main()