# Binary encoding

Status: **design note — not yet implemented.**

Today `llmcodec` encodes a sentence to a single real number in `[0, 1]` and emits
the **midpoint** of the final interval. This document specifies an additional,
complementary output: a **binary code**, i.e. the bit string you get by walking
down the dyadic grid until one cell is fully contained in the final interval.

Everything below is verified against the current implementation; the numbers in
the worked example are reproducible with the code in `src/llmcodec/interval.py`.

## 1. Motivation

The real number the codec produces is an idealised object: it carries an
infinite-precision payload and cannot be serialised as-is. A binary code is the
*concrete* form of the same information, and it makes three things explicit that
are currently implicit:

- **How many bits does a sentence cost?** The answer falls out of the interval
  width, and it is within a couple of bits of the theoretical bound.
- **Where does the "arithmetic coding" name come from?** The dyadic walk *is*
  binary arithmetic coding, in its most direct form.
- **Why exact rational arithmetic?** At 40–65 bits a `float` would silently
  corrupt the code. The existing `Fraction` representation is what makes this
  correct.

## 2. The rule

Let the final interval be `[lo, hi]`, with `lo, hi` exact `Fractions`.

For `n = 1, 2, 3, ...`, the unit interval is cut into `2^n` **dyadic cells**

```
[0/2^n, 1/2^n], [1/2^n, 2/2^n], ..., [(2^n - 1)/2^n, 2^n/2^n]
```

and we pick the leftmost cell that is **entirely contained** in `[lo, hi]`:

```
k = ceil(lo * 2^n)                      # smallest integer with k/2^n >= lo
contained  <=>  k/2^n >= lo  and  (k + 1)/2^n <= hi
```

Stop at the smallest `n` satisfying `contained`. The code is then `k` written as
exactly `n` bits (zero-padded on the left).

### 2.1 Why "otherwise 1" is not the whole story

The intuitive description — "if the range is inside `[0, 0.5)` emit `0`,
otherwise `1`" — is correct *only when the interval already fits inside one
half*. When the interval **straddles** a boundary, no bit can be committed yet
and the algorithm must move to the next level without emitting anything.

Concrete example, interval `[3/10, 7/10]` (straddles `0.5`):

| n | k | cell | contained? |
|---|---|---|---|
| 1 | 1 | `[1/2, 1]` | no — cell extends past `hi = 0.7` |
| 2 | 2 | `[1/2, 3/4]` | no — `3/4 > 0.7` |
| 3 | 3 | `[3/8, 1/2]` | **yes** → bits `011` |

So the "greedy containment" formulation is the correct one, and it subsumes the
"emit a bit per level" intuition: a bit is only determined once containment
holds.

### 2.2 Containment implies `2^-n <= width`

Containment requires a cell of width `1/2^n` to fit inside an interval of width
`w = hi - lo`, so necessarily `1/2^n <= w`, i.e.

```
n >= ceil(-log2(w))
```

This gives the lower bound on the code length. Exhaustive search over small
rational intervals (`Interval(a, b, d)` for `d < 25`) found **zero** cases where
a contained cell existed with `2^-n > w`, confirming the bound.

## 3. Worked example (fully hand-checkable)

Take a two-symbol source with weights `6:4` — no model, no vocabulary, just
`Interval.subdivide`. Encode `A` then `B`.

```
start:      [0, 1]
A (index 0): subdivide([6,4], 0) -> [0, 3/5]        width 3/5
B (index 1): subdivide([6,4], 1) -> [9/25, 3/5]     width 6/25
```

Final interval `[9/25, 15/25]`, width `6/25 = 0.24`, so
`ceil(-log2(0.24)) = 3` is the minimum possible length.

| n | k = ceil(lo·2^n) | cell | contained? |
|---|---|---|---|
| 1 | 1 | `[1/2, 1]` | no |
| 2 | 2 | `[1/2, 3/4]` | no |
| 3 | 3 | `[3/8, 1/2]` | **yes** |

Code: `n = 3`, `k = 3` → bits **`011`**, value `3/8 = 0.375` (which indeed lies
in `[0.36, 0.6]`).

Decoding `011` walks the same tree:

```
[0,1].find_slice([6,4], 3/8) = 0   -> 'A'
[0,3/5].find_slice([6,4], 3/8) = 1 -> 'B'
```

3 bits for 2 symbols at ~2.06 bits of entropy — comfortably within the
arithmetic-coding guarantee.

## 4. Measured behaviour on real sentences

Encoded with an untrained tiny model and the lossless `full` selector
(`d_model=16, n_layers=1`), comparing the optimal length `ceil(-log2(w))` with
the length actually needed:

| text | interval width | `ceil(-log2 w)` | n needed | bits |
|---|---|---|---|---|
| `'a'` | 5.51e-04 | 11 | **12** | `000010101011` |
| `'zebra.'` | 5.37e-12 | 37 | **38** | `10101001001101111110001101000010001010` |
| `'the cat'` | 1.74e-13 | 43 | **44** | `10000100011010100011111011110100011011110001` |
| `'hello world'` | 4.28e-20 | 65 | **65** | `00110110011010100101001100001100110101010110111100110110000101011` |

The overshoot is 0–2 bits: the cost of restricting the code to a *closed* dyadic
cell rather than an arbitrary point in the interval. This overhead is bounded
and independent of sentence length, which is exactly the classic result.

## 5. Correctness argument

The only real hazard is that `k/2^n` may land **exactly** on a boundary between
two candidate slices inside the interval, where a naive half-open convention
would pick the wrong slice.

This is safe, and here is why:

1. **`k/2^n` is never equal to `hi`.** If `k/2^n == hi` then the cell's upper
   endpoint `(k+1)/2^n` would exceed `hi`, contradicting containment. So the
   code always lies strictly below the right edge. (Exhaustive search over small
   rational intervals: **0** counterexamples.)
2. **Landing exactly on an internal boundary is harmless.** `Interval.find_slice`
   treats slices as half-open `[a, b)` with the final slice closed on both ends,
   so a value equal to an internal boundary resolves to the slice on the
   **right** — which is precisely the slice the encoder chose. (Exhaustive search
   found **68** such boundary cases; all 68 decoded to the correct slice.)
3. **Endpoints must stay `Fraction`.** At 65 bits, `2^-65` is far below float
   precision. Conversion to `float` must happen only for display, never for
   code selection or decoding.

## 6. Proposed API

The math belongs in `interval.py`, where it is testable with no model at all.

```python
class Interval:
  def dyadic(self):
    """Return (n, k): smallest n and the k with 2^-n cell k inside self."""
    # k = ceil(lo * 2^n), first n where k/2^n >= lo and (k+1)/2^n <= hi
```

```python
class BitCode(namedtuple("BitCode", "n k text")):
  @property
  def bits(self):   # zero-padded bit string of length n
  @property
  def value(self):  # Fraction(k, 2**n) -- exact, never float
```

```python
class Codec:
  def encode_bits(self, text):   # -> BitCode
  def decode_bits(self, n, k):   # -> str; decodes Fraction(k, 2**n)
```

`Encoded` gains a convenience accessor delegating to `Interval.dyadic()`.

## 7. Open decision: is the bit string self-delimiting?

**It is not.** `011` and `0011` are different messages, so `n` is part of the
payload and must travel with the bits (hence `BitCode` carrying both `n` and
`k`).

Three ways forward, to be decided before implementation:

- **(a) Length-prefixed.** Emit `n` followed by the `n` bits. Simple and
  explicit; costs `O(log n)` extra bits.
- **(b) Deferred / streaming decode.** Do not transmit `n`; have the receiver
  consume bits until it has narrowed to an interval that provably contains the
  answer, then stop. This is what real arithmetic coders do, and it removes the
  length overhead — but it needs the decoder to widen its acceptance window as
  more bits arrive, which is a genuinely more subtle algorithm.
- **(c) Fixed-length API only.** `encode_bits(text, n_bits)` for callers that
  know the budget, with greedy `dyadic()` used internally for validation.

Recommendation for a first implementation: **(a)**, because it is provably
correct with almost no new machinery, and it keeps the decoder trivial. (b)
makes an excellent follow-up contribution once (a) is covered by tests.

## 8. Test plan

No model needed for the majority of these — a fixed `Interval` suffices:

- `Interval.unit().dyadic()` → `(1, 0)` (bits `0`, cell `[0, 1/2]`).
- Hand example from section 3: `Interval(9, 15, 25).dyadic() == (3, 3)` and the
  bits are `011`.
- Round-trip: for a lattice of small rational intervals, `decode_bits(encode)
  == original` and the code's value lies inside the interval.
- **Straddle case:** `[3/10, 7/10]` must skip `n = 1, 2` and land on `n = 3`.
- **Boundary case:** at least one interval whose code lands exactly on an
  internal slice boundary decodes correctly (regression guard for §5.2).
- **Length bound:** `n >= ceil(-log2(width))` for every case.
- **Float hazard:** a 65-bit code round-trips through `Fraction` while the same
  value as `float` is demonstrably lossy.

## 9. Non-goals

- No change to the existing `encode`/`decode` real-number API.
- No bit-level packing across multiple messages.
- No change to `WEIGHT_SCALE` or the selector strategy.
