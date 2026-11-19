"""Exact rational interval arithmetic for the codec.

The codec represents the current probability interval as an exact rational
interval ``[num_lo / den, num_hi / den]`` rather than a pair of floating point
numbers.  Floating point rounding would accumulate over the many subdivision
steps of a sentence and let the interval recovered during decoding drift away
from the one produced during encoding; exact arithmetic removes that failure
mode entirely, so encoding and decoding are perfectly inverse.

Each coding step splits the current interval into one slice per candidate token,
with slice widths proportional to the (integer) candidate weights.  Because the
weights are arbitrary integers, the slices are generally *unequal*, so
subdivision cannot be expressed with a single numerator.  Instead,
:meth:`Interval.subdivide` takes the full list of weights and the index of the
desired slice, and computes the slice boundaries from the normalized cumulative
weights.

Every interval is stored in lowest terms: the constructor reduces the triple
``(num_lo, num_hi, den)`` by their greatest common divisor.  This keeps the
integers small without changing the represented interval, which matters because
the denominator otherwise grows by a factor of ``sum(weights)`` at every step.
"""

from collections import namedtuple
from fractions import Fraction
from math import gcd


class BitCode(namedtuple("BitCode", "n k")):
  """A dyadic-cell binary code: ``n`` bits whose value is ``k / 2**n``.

  A :class:`BitCode` is the *concrete* form of an :class:`Interval`.  It names
  the dyadic cell ``[k / 2**n, (k + 1) / 2**n]`` -- a sub-interval of ``[0, 1]``
  whose endpoints are multiples of ``2**-n`` -- that is fully contained in the
  interval it was derived from.  Because the whole cell lies inside the source
  interval, *every* point of the cell is a valid code for the same sentence, so
  the code is safe to store and decode without any rounding caveats.

  The two fields are the only state:

  Attributes:
    n: The number of bits, i.e. the grid depth.  ``n == 0`` only for the unit
      interval ``[0, 1]`` (see :meth:`Interval.dyadic`).
    k: The cell index in ``[0, 2**n]``.  The cell is ``[k / 2**n, (k +
      1) / 2**n]``.

  Examples:
    >>> BitCode(3, 3).bits
    '011'
    >>> BitCode(3, 3).value
    Fraction(3, 8)
    >>> BitCode(3, 3).width
    Fraction(1, 8)
    >>> BitCode(3, 3).cell.lower
    Fraction(3, 8)
  """

  __slots__ = ()

  @property
  def bits(self):
    """The code as a zero-padded bit string of exactly ``n`` characters.

    The string is ``k`` written in binary with ``n`` digits; leading zeros are
    significant because they encode the grid depth ``n``.  The unit interval --
    the only source of ``n == 0`` -- yields the empty string.

    Examples:
      >>> BitCode(3, 3).bits
      '011'
      >>> BitCode(0, 0).bits
      ''
    """
    return format(self.k, f"0{self.n}b") if self.n else ""

  @property
  def value(self):
    """The lower edge of the cell as an exact :class:`~fractions.Fraction`.

    This is a point *inside* the source interval (it is at or above its lower
    bound), so it is a valid code for the sentence.  It is always exact; never
    convert it to ``float`` for storage or decoding.

    Examples:
      >>> BitCode(3, 3).value
      Fraction(3, 8)
    """
    return Fraction(self.k, 1 << self.n)

  @property
  def width(self):
    """The width of the cell, ``2**-n``, as an exact :class:`~fractions.Fraction`.

    Examples:
      >>> BitCode(3, 3).width
      Fraction(1, 8)
    """
    return Fraction(1, 1 << self.n)

  @property
  def cell(self):
    """The dyadic cell itself, as an :class:`Interval`.

    The cell ``[k / 2**n, (k + 1) / 2**n]`` is contained in the interval the
    code was derived from, and any point inside it decodes to the same sentence.

    Examples:
      >>> BitCode(3, 3).cell.lower
      Fraction(3, 8)
      >>> BitCode(3, 3).cell.upper
      Fraction(1, 2)
    """
    return Interval(self.k, self.k + 1, 1 << self.n)


class Interval(namedtuple("Interval", "num_lo num_hi den")):
  """A closed rational interval ``[num_lo / den, num_hi / den]``.

  The interval is immutable and normalized to lowest terms.  All three fields
  are integers and satisfy ``den > 0`` and ``0 <= num_lo <= num_hi <= den``, so
  every interval lies inside ``[0, 1]``.

  Attributes:
    num_lo: Numerator of the (inclusive) lower bound.
    num_hi: Numerator of the (inclusive) upper bound.
    den: Common denominator, strictly positive.
  """

  __slots__ = ()

  def __new__(cls, num_lo, num_hi, den):
    """Validate the invariants and build the interval.

    Raises:
      ValueError: If the denominator is not positive, the bounds are inverted,
        or the interval falls outside ``[0, 1]``.
    """
    if den <= 0:
      raise ValueError("denominator must be positive")
    if num_lo > num_hi:
      raise ValueError("lower bound must not exceed upper bound")
    if num_lo < 0 or num_hi > den:
      raise ValueError("interval must lie inside [0, 1]")
    return super().__new__(cls, num_lo, num_hi, den)

  @classmethod
  def unit(cls):
    """Return the full interval ``[0, 1]``.

    This is the starting interval for both encoding and decoding.

    Example:
      >>> Interval.unit()
      Interval([0.000000, 1.000000] = 0/1/1)
    """
    return cls(0, 1, 1)

  @classmethod
  def _reduce(cls, num_lo, num_hi, den):
    """Construct an interval reduced to lowest terms.

    Args:
      num_lo: Numerator of the lower bound.
      num_hi: Numerator of the upper bound.
      den: Common denominator.

    Returns:
      The equivalent interval with ``gcd(num_lo, num_hi, den)`` divided out.
    """
    g = gcd(gcd(num_lo, num_hi), den)
    if g > 1:
      num_lo, num_hi, den = num_lo // g, num_hi // g, den // g
    return cls(num_lo, num_hi, den)

  @classmethod
  def from_fraction(cls, value, den):
    """Return the unit interval of width ``1 / den`` that contains ``value``.

    This is used to *quantize* an arbitrary real back onto the codec's grid,
    for example when a caller supplies a code as a plain float or fraction.

    Args:
      value: A rational number in ``[0, 1]``.
      den: The resolution (number of unit intervals across ``[0, 1]``).

    Returns:
      The largest interval of width ``1 / den`` whose interior contains
      ``value``.  A value of exactly ``1`` maps to the last unit interval.

    Raises:
      ValueError: If ``value`` lies outside ``[0, 1]``.
    """
    if not (0 <= value <= 1):
      raise ValueError("value must lie in [0, 1]")
    scaled = value * den
    k = int(scaled)  # floor of value * den
    if k == den:  # value == 1 -> the last unit interval
      k = den - 1
    return cls(k, k + 1, den)

  @property
  def lower(self):
    """The inclusive lower bound as a :class:`~fractions.Fraction`."""
    return Fraction(self.num_lo, self.den)

  @property
  def upper(self):
    """The inclusive upper bound as a :class:`~fractions.Fraction`."""
    return Fraction(self.num_hi, self.den)

  @property
  def width(self):
    """The width ``upper - lower`` as a :class:`~fractions.Fraction`.

    The width is a natural measure of how "uncertain" the code still is: a
    wider interval means fewer bits of information have been committed, which is
    what a well-trained language model achieves for likely sentences.

    Example:
      >>> Interval(0, 3, 4).width
      Fraction(3, 4)
    """
    return Fraction(self.num_hi - self.num_lo, self.den)

  def midpoint(self):
    """Return the center of the interval.

    The encoder emits this value as the final code.  Any point strictly inside
    the interval decodes to the same sentence, so the midpoint is a robust
    choice that stays as far as possible from the boundaries.

    Example:
      >>> Interval(0, 3, 4).midpoint()
      Fraction(3, 8)
    """
    return Fraction(self.num_lo + self.num_hi, 2 * self.den)

  def contains(self, value):
    """Return whether ``value`` lies within the closed interval.

    Example:
      >>> Interval(0, 3, 4).contains(Fraction(1, 2))
      True
      >>> Interval(0, 3, 4).contains(Fraction(7, 8))
      False
    """
    return self.lower <= value <= self.upper

  def dyadic(self):
    """Return the smallest :class:`BitCode` cell contained in the interval.

    Cut ``[0, 1]`` into ``2**n`` equal *dyadic cells* for ``n = 0, 1, 2, ...``
    and pick the coarsest cell that is **entirely contained** in ``self``.  The
    result is a :class:`BitCode` ``(n, k)`` naming the cell ``[k / 2**n,
    (k + 1) / 2**n]``.

    The cell is contained, so any point of it -- in particular the code's own
    :attr:`~BitCode.value` ``k / 2**n`` -- lies inside ``self`` and therefore
    decodes to the same sentence.  This is what makes the binary code safe: it
    never relies on staying clear of an interval boundary.

    Containment needs a cell of width ``2**-n`` to fit in a width ``w = hi -
    lo``, so ``n >= ceil(-log2(w))``; the returned ``n`` is the smallest such
    value (0 to 2 bits above the information-theoretic optimum).  The unit
    interval returns ``BitCode(0, 0)``; the width of ``self`` must be positive.

    Returns:
      The :class:`BitCode` for the coarsest contained dyadic cell.

    Raises:
      ValueError: If the interval has zero width, so no cell can be contained.

    Examples:
      >>> Interval(9, 15, 25).dyadic()
      BitCode(n=3, k=3)
      >>> Interval(9, 15, 25).dyadic().bits
      '011'
      >>> Interval(0, 1, 1).dyadic()
      BitCode(n=0, k=0)
    """
    span = self.num_hi - self.num_lo
    if span <= 0:
      raise ValueError("a zero-width interval contains no dyadic cell")
    # Smallest n with 2**n >= den / span, i.e. cell width 1 / 2**n <= span.
    # The closed form lands on the answer or one short; the loop settles it.
    n = (-(-self.den // span) - 1).bit_length()
    while True:
      step = 1 << n
      # k = ceil(lo * 2**n), the left edge of the leftmost candidate cell.
      k = -(-(self.num_lo * step) // self.den)
      if (k + 1) * self.den <= self.num_hi * step:
        return BitCode(n, k)
      n += 1

  def subdivide(self, counts, index):
    """Return the ``index``-th slice of ``self`` under integer ``counts``.

    The slice boundaries sit at the normalized cumulative weights: with total
    weight ``T = sum(counts)``, slice ``i`` spans ``[p_i / T, p_{i+1} / T]`` of
    ``self``, where ``p_i = sum(counts[:i])``.  This is exactly the arithmetic
    coding step that consumes the probability mass assigned to the observed
    token.

    Args:
      counts: Non-negative integer weights, one per candidate token.
      index: Which slice to return.

    Returns:
      The selected sub-interval, reduced to lowest terms.

    Raises:
      ValueError: If ``counts`` is empty, contains a negative weight, sums to
        zero, or if ``index`` is out of range.

    Example:
      Subdividing ``[0, 1]`` into a 3:1 split puts the first branch in the
      lower three quarters:

      >>> Interval.unit().subdivide([3, 1], 0)
      Interval([0.000000, 0.750000] = 0/3/4)
      >>> Interval.unit().subdivide([3, 1], 1)
      Interval([0.750000, 1.000000] = 3/4/4)
    """
    if not counts:
      raise ValueError("counts must be non-empty")
    if any(c < 0 for c in counts):
      raise ValueError("counts must be non-negative")
    total = sum(counts)
    if total <= 0:
      raise ValueError("total weight must be positive")
    if not (0 <= index < len(counts)):
      raise ValueError(f"index {index} out of range for {len(counts)} slices")

    prefix = sum(counts[:index])
    span = self.num_hi - self.num_lo
    lo = self.num_lo * total + prefix * span
    hi = self.num_lo * total + (prefix + counts[index]) * span
    return Interval._reduce(lo, hi, self.den * total)

  def find_slice(self, counts, value):
    """Return the index of the slice of ``self`` that contains ``value``.

    This is the inverse of :meth:`subdivide`: it is what the decoder uses to
    recover the token from a code.  Slices are treated as half-open on the left
    and closed on the right so that the mapping matches the encoder's
    cumulative-weight lookup, and the final slice is closed on both ends so that
    ``value == upper`` always resolves.

    Args:
      counts: Non-negative integer weights, one per candidate token.
      value: A rational number inside ``self``.

    Returns:
      The index of the slice containing ``value``.

    Raises:
      ValueError: If ``value`` lies outside ``self`` or ``self`` has zero width.

    Example:
      A 3:1 split of ``[0, 1]`` sends ``7/8`` to the upper branch:

      >>> Interval.unit().find_slice([3, 1], Fraction(7, 8))
      1
      >>> Interval.unit().find_slice([3, 1], Fraction(1, 8))
      0
    """
    if not self.contains(value):
      raise ValueError("value lies outside the interval")
    total = sum(counts)
    span = self.num_hi - self.num_lo
    if span == 0:
      raise ValueError("cannot subdivide a zero-width interval")

    # Position of ``value`` within ``self`` measured in units of ``1 / total``.
    pos = (value - self.lower) * total / self.width
    cumulative = 0
    for i, c in enumerate(counts):
      nxt = cumulative + c
      if pos < nxt or i == len(counts) - 1:
        return i
      cumulative = nxt
    return len(counts) - 1  # pragma: no cover - defensive

  def __repr__(self):  # pragma: no cover - cosmetic
    lo, hi = float(self.lower), float(self.upper)
    return f"Interval([{lo:.6f}, {hi:.6f}] = {self.num_lo}/{self.num_hi}/{self.den})"
