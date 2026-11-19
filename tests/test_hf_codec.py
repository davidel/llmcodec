"""End-to-end codec tests over a Hugging Face model and tokenizer.

Builds a tiny random GPT-2 and a word-level fast tokenizer in memory, so
nothing is downloaded and the tests run offline.  The HF pair plays both
roles -- distribution source and vocabulary -- exactly as
``--hf-model ID --tokenizer ID`` would at the CLI.
"""

from argparse import Namespace

import pytest

from llmcodec.candidates import build_selector
from llmcodec.codec import Codec, NotEncodableError
from llmcodec.pretrained import HFCausalLM, HFTokenizer, TokenCodec
from llmcodec.vocab import DEFAULT_VOCAB

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
WORDS = ["the", "cat", "sat", "on", "mat", "a", "dog", "ran", "big", "small"]
VOCAB_SIZE = len(SPECIALS) + len(WORDS)


def _word_level_tokenizer(with_bos=True):
  """Build a fast word-level tokenizer with a fixed toy vocabulary."""
  pytest.importorskip("transformers")
  from tokenizers import Tokenizer, models, pre_tokenizers
  from transformers import PreTrainedTokenizerFast

  vocab = {tok: i for i, tok in enumerate(SPECIALS + WORDS)}
  backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
  backend.pre_tokenizer = pre_tokenizers.Whitespace()
  # No decoder component: the WordLevel backend already re-joins words with
  # single spaces, while attaching an explicit decoder glues them together.
  specials = {"eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>"}
  if with_bos:
    specials["bos_token"] = "<bos>"
  return HFTokenizer(PreTrainedTokenizerFast(tokenizer_object=backend, **specials),
                     name="toy-wordlevel")


def _hf_model(vocab_size=VOCAB_SIZE):
  """Build a tiny random GPT-2 whose vocabulary matches the tokenizer."""
  pytest.importorskip("transformers")
  from transformers import GPT2Config, GPT2LMHeadModel

  config = GPT2Config(vocab_size=vocab_size,
                      n_positions=16,
                      n_embd=32,
                      n_layer=2,
                      n_head=4)
  return HFCausalLM(GPT2LMHeadModel(config),
                    name="test-gpt2",
                    dtype="float32",
                    device="cpu")


def test_tokencodec_protocol_adoption():
  """Both the built-in vocab and the HF wrapper satisfy the protocol."""
  assert isinstance(DEFAULT_VOCAB, TokenCodec)
  assert isinstance(_word_level_tokenizer(), TokenCodec)


def test_hf_tokenizer_encode_adds_no_specials():
  """``encode`` returns content ids only; the codec manages BOS/EOS itself."""
  tok = _word_level_tokenizer()
  ids = tok.encode("the cat sat")
  word_ids = {t: i for i, t in enumerate(SPECIALS + WORDS)}
  assert ids == [word_ids[w] for w in ("the", "cat", "sat")]
  assert tok.bos_id not in ids
  assert tok.eos_id not in ids


def test_unknown_word_raises_under_topn_like_char_level():
  """``topn`` raises for out-of-vocabulary inputs, exactly as char-level."""
  tok = _word_level_tokenizer()
  codec = Codec(_hf_model(), tok, build_selector("topn", n=8), device="cpu")
  with pytest.raises(NotEncodableError, match="unk"):
    codec.encode("the zebra")


def test_unknown_word_encodable_under_adaptive():
  """``adaptive`` grows its candidate set until ``<unk>`` is covered."""
  tok = _word_level_tokenizer()
  codec = Codec(_hf_model(),
                tok,
                build_selector("adaptive", max_n=16, coverage=0.999),
                device="cpu")
  enc = codec.encode("the zebra")
  assert codec.decode(enc.value) == "the <unk>"


def test_hf_tokenizer_decode_roundtrip():
  """Ids decode back to the original words."""
  tok = _word_level_tokenizer()
  ids = tok.encode("the cat sat")
  assert tok.decode(ids) == "the cat sat"


def test_hf_tokenizer_specials_and_itos():
  """Special ids and the ``itos`` table come from the wrapped tokenizer."""
  tok = _word_level_tokenizer()
  assert tok.bos_id == SPECIALS.index("<bos>")
  assert tok.eos_id == SPECIALS.index("<eos>")
  assert tok.unk_id == SPECIALS.index("<unk>")
  assert tok.vocab_size == VOCAB_SIZE
  assert len(tok.itos) == VOCAB_SIZE
  assert tok.itos[tok.unk_id] == "<unk>"


def test_hf_tokenizer_encode_bos():
  """``add_bos`` prepends the BOS id."""
  tok = _word_level_tokenizer()
  assert tok.encode("the", add_bos=True)[0] == tok.bos_id


def test_hf_tokenizer_no_bos_rejects_add_bos():
  """Requesting BOS on a tokenizer without one fails loudly."""
  tok = _word_level_tokenizer(with_bos=False)
  assert tok.bos_id is None
  with pytest.raises(ValueError, match="no bos token"):
    tok.encode("the", add_bos=True)


def test_hf_tokenizer_decode_skips_specials():
  """``skip_special`` drops BOS/EOS/UNK ids before decoding."""
  tok = _word_level_tokenizer()
  ids = [tok.bos_id] + tok.encode("the cat") + [tok.eos_id]
  assert tok.decode(ids, skip_special=True) == "the cat"


def test_codec_with_hf_roundtrip_is_exact():
  """Adaptive-mode roundtrip through the HF model and tokenizer is exact."""
  tok = _word_level_tokenizer()
  codec = Codec(_hf_model(),
                tok,
                build_selector("adaptive", max_n=16, coverage=0.999),
                device="cpu")
  text = "the cat sat on a mat"
  assert codec.decode(codec.encode(text).value) == text


def test_codec_with_hf_bits_roundtrip_is_exact():
  """The dyadic BitCode path roundtrips exactly over the HF pair."""
  tok = _word_level_tokenizer()
  codec = Codec(_hf_model(),
                tok,
                build_selector("adaptive", max_n=16, coverage=0.999),
                device="cpu")
  text = "a big dog ran"
  code = codec.encode_bits(text)
  assert codec.decode_bits(code) == text


def test_subword_emittable_ids():
  """Emittable ids are everything except BOS and PAD, unlike char-level."""
  tok = _word_level_tokenizer()
  codec = Codec(_hf_model(), tok, build_selector("topn", n=8), device="cpu")
  pad_id = SPECIALS.index("<pad>")
  expected = set(range(VOCAB_SIZE)) - {tok.bos_id, pad_id}
  assert set(codec.emittable_ids) == expected
  assert tok.eos_id in codec.emittable_ids
  assert tok.unk_id in codec.emittable_ids


def test_codec_rejects_model_tokenizer_mismatch():
  """A model/tokenizer size mismatch fails at construction, not mid-decode."""
  tok = _word_level_tokenizer()
  with pytest.raises(ValueError, match="vocabulary size"):
    Codec(_hf_model(vocab_size=VOCAB_SIZE * 2),
          tok,
          build_selector("topn", n=8),
          device="cpu")


def test_load_codec_model_requires_exactly_one_source():
  """``--model`` and ``--hf-model`` are mutually exclusive but one is needed."""
  from llmcodec.cli import _load_codec_model

  with pytest.raises(ValueError, match="mutually exclusive"):
    _load_codec_model(Namespace(model="ckpt.pt", hf_model="gpt2", device="cpu"))
  with pytest.raises(ValueError, match="one of --model or --hf-model"):
    _load_codec_model(Namespace(model=None, hf_model=None, device="cpu"))
