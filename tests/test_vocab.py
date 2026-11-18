from llmcodec.vocab import CHARS, DEFAULT_VOCAB, SPECIAL_TOKENS, Vocab


def test_vocab_size_and_order():
    assert DEFAULT_VOCAB.vocab_size == len(CHARS) + len(SPECIAL_TOKENS)
    assert DEFAULT_VOCAB.vocab_size == 43  # 39 chars + 4 specials
    # specials occupy the first ids, characters follow in natural order
    assert DEFAULT_VOCAB.itos[:4] == SPECIAL_TOKENS
    assert DEFAULT_VOCAB.itos[4:] == tuple(CHARS)


def test_special_token_ids():
    v = DEFAULT_VOCAB
    assert v.pad_id == 0
    assert v.bos_id == 1
    assert v.eos_id == 2
    assert v.unk_id == 3


def test_encode_lowercases():
    v = DEFAULT_VOCAB
    assert v.encode("AbC") == v.encode("abc")


def test_encode_unknown_becomes_unk():
    v = DEFAULT_VOCAB
    ids = v.encode("a!b")
    assert v.unk_id in ids
    assert len(ids) == 3


def test_encode_decode_roundtrip():
    v = DEFAULT_VOCAB
    text = "hello, world. 123"
    assert v.decode(v.encode(text)) == text


def test_encode_add_bos_eos():
    v = DEFAULT_VOCAB
    ids = v.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == v.bos_id
    assert ids[-1] == v.eos_id


def test_decode_skip_special():
    v = DEFAULT_VOCAB
    ids = [v.bos_id] + v.encode("hi") + [v.eos_id]
    assert v.decode(ids, skip_special=True) == "hi"
    assert v.decode(ids) == "<bos>hi<eos>"


def test_serialization_roundtrip():
    v = DEFAULT_VOCAB
    assert Vocab.from_dict(v.to_dict()) == v


def test_build_rejects_duplicates():
    import pytest

    with pytest.raises(ValueError):
        Vocab.build(chars="aab")
