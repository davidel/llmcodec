import numpy as np

from llmcodec.candidates import (
    AdaptiveSelector,
    CandidateSet,
    FullVocabSelector,
    TopNSelector,
    build_selector,
)

ELIGIBLE = list(range(10))


def _probs(values):
  p = np.zeros(20, dtype=np.float64)
  for i, v in enumerate(values):
    p[i] = v
  return p


def test_candidate_set_validation():
  import pytest

  with pytest.raises(ValueError):
    CandidateSet((1, 2), (1,))
  with pytest.raises(ValueError):
    CandidateSet((), ())
  with pytest.raises(ValueError):
    CandidateSet((1, 2), (0, 3))


def test_full_selector_keeps_all_nonzero_sorted():
  p = _probs([0.0, 0.2, 0.0, 0.3, 0.5])
  cs = FullVocabSelector().select(p, ELIGIBLE)
  assert cs.token_ids == (1, 3, 4)
  assert all(w > 0 for w in cs.weights)


def test_topn_selector_takes_most_probable_then_sorts():
  p = _probs([0.05, 0.1, 0.6, 0.2, 0.05])
  cs = TopNSelector(n=2).select(p, ELIGIBLE)
  # most probable are ids 2 (0.6) and 3 (0.2); laid out sorted
  assert cs.token_ids == (2, 3)


def test_topn_ignores_zero_probability():
  p = _probs([0.0, 0.0, 1.0, 0.0])
  cs = TopNSelector(n=5).select(p, ELIGIBLE)
  assert cs.token_ids == (2,)


def test_adaptive_grows_to_coverage():
  p = _probs([0.5, 0.25, 0.15, 0.05, 0.05])
  cs = AdaptiveSelector(max_n=16, coverage=0.9).select(p, ELIGIBLE)
  assert cs.token_ids == (0, 1, 2)  # 0.5 + 0.25 + 0.15 >= 0.9


def test_adaptive_respects_max_n():
  p = _probs([0.2, 0.2, 0.2, 0.2, 0.2])
  cs = AdaptiveSelector(max_n=2, coverage=0.99).select(p, ELIGIBLE)
  assert len(cs) == 2


def test_build_selector_modes():
  assert isinstance(build_selector("full"), FullVocabSelector)
  assert isinstance(build_selector("topn", n=3), TopNSelector)
  assert isinstance(build_selector("adaptive"), AdaptiveSelector)
