from src.data.tokenizer import BPETokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = BPETokenizer.train(["hello world", "hello moe"], max_vocab_size=32, min_frequency=1)
    encoded = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    assert "hello" in decoded.lower()
    assert "world" in decoded.lower()

