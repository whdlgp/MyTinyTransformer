from tokenizer.simple_tokenizer import Tokenizer


if __name__ == "__main__":
    text = "hello world"

    tokenizer = Tokenizer.from_text(text)

    print("vocab size:", tokenizer.vocab_size)
    print("vocab     :", tokenizer.state_dict()["chars"])

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("input     :", text)
    print("encoded   :", encoded)
    print("decoded   :", decoded)

    assert decoded == text
    print("encode/decode: OK")

    state = tokenizer.state_dict()

    tokenizer2 = Tokenizer.from_state_dict(state)

    encoded2 = tokenizer2.encode(text)
    decoded2 = tokenizer2.decode(encoded2)

    assert encoded2 == encoded
    assert decoded2 == text
    assert tokenizer2.vocab_size == tokenizer.vocab_size

    print("state_dict: OK")
    print("restore   : OK")