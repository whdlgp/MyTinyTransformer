class Tokenizer:
    def __init__(self, chars):
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"chars": list(self.itos[i] for i in range(self.vocab_size))}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["chars"])