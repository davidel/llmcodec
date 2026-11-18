"""Exact rational interval arithmetic used by the codec.

The codec represents the current probability interval as an exact rational
interval ``[num_lo / den, num_hi / den]`` rather than a pair of floats.  This
keeps encoding/decoding *exactly* invertible no matter how deep the recursion
goes: floating point rounding would otherwise make the recovered interval drift
away from the encoded one.

Sub-slices have widths proportional to the (normalized) next-token
probabilities, so they are generally *not* equal.  Subdivision therefore takes
a list of integer *counts* (one per candidate token); the ``i``-th slice spans
the fraction ``[prefix_i / total, prefix_{i+1} / total]`` of the current
interval, where ``prefix`` are the cumulative counts and ``total`` their sum.

Every constructor reduces the fraction by the gcd of its three integers, which
keeps the numbers from growing without bound while preserving the represented
interval exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd


@dataclass(frozen=True)
class Interval:
    """The closed rational interval ``[num_lo / den, num_hi / den]``.

    Invariants: ``den > 0`` and ``0 <= num_lo <= num_hi <= den``.
    """

    num_lo: int
    num_hi: int
    den: int

    def __post_init__(self) -> None:
        if self.den <= 0:
            raise ValueError("denominator must be positive")
        if self.num_lo > self.num_hi:
            raise ValueError("lower bound must not exceed upper bound")
        if self.num_lo < 0 or self.num_hi > self.den:
            raise ValueError("interval must lie inside [0, 1]")

    # -- constructors --------------------------------------------------------
    @classmethod
    def unit(cls) -> Interval:
        """The full ``[0, 1]`` interval."""
        return cls(0, 1, 1)

    @classmethod
    def _make(cls, num_lo: int, num_hi: int, den: int) -> Interval:
        """Build a reduced interval."""
        g = gcd(gcd(num_lo, num_hi), den)
        if g > 1:
            num_lo, num_hi, den = num_lo // g, num_hi // g, den // g
        return cls(num_lo, num_hi, den)

    @classmethod
    def from_fraction(cls, value: Fraction, den: int) -> Interval:
        """Build the unit interval (of width ``1/den``) containing ``value``.

        The result is the largest interval of the requested resolution that
        contains ``value``.
        """
        if not (0 <= value <= 1):
            raise ValueError("value must lie in [0, 1]")
        scaled = value * den
        k = int(scaled)  # floor
        if k == den:  # value == 1 -> the last unit interval
            k = den - 1
        return cls(k, k + 1, den)

    # -- value accessors -----------------------------------------------------
    @property
    def lower(self) -> Fraction:
        return Fraction(self.num_lo, self.den)

    @property
    def upper(self) -> Fraction:
        return Fraction(self.num_hi, self.den)

    @property
    def width(self) -> Fraction:
        return Fraction(self.num_hi - self.num_lo, self.den)

    def midpoint(self) -> Fraction:
        """The center of the interval -- the value emitted when encoding."""
        return Fraction(self.num_lo + self.num_hi, 2 * self.den)

    def contains(self, value: Fraction) -> bool:
        return self.lower <= value <= self.upper

    # -- subdivision ---------------------------------------------------------
    def subdivide(self, counts: Sequence[int], index: int) -> Interval:
        """Return the ``index``-th slice of ``self`` under integer ``counts``.

        ``counts[i]`` is the (integer) weight of candidate ``i``; the slice
        boundaries sit at the normalized cumulative weights.
        """
        if not counts:
            raise ValueError("counts must be non-empty")
        if any(c < 0 for c in counts):
            raise ValueError("counts must be non-negative")
        if sum(counts) <= 0:
            raise ValueError("total weight must be positive")
        if not (0 <= index < len(counts)):
            raise ValueError(f"index {index} out of range for {len(counts)} slices")

        total = sum(counts)
        prefix = sum(counts[:index])
        span = self.num_hi - self.num_lo
        lo = self.num_lo * total + prefix * span
        hi = self.num_lo * total + (prefix + counts[index]) * span
        return Interval._make(lo, hi, self.den * total)

    def find_slice(self, counts: Sequence[int], value: Fraction) -> int:
        """Return the index of the slice containing ``value``.

        Slices are treated as half-open ``[boundary_i, boundary_{i+1})`` so the
        mapping agrees with the encoder's cumulative-weight lookup.  The final
        slice is closed on the right so that ``value == 1`` is always resolved.
        """
        if not self.contains(value):
            raise ValueError("value lies outside the interval")
        total = sum(counts)
        span = self.num_hi - self.num_lo
        if span == 0:
            raise ValueError("cannot subdivide a zero-width interval")

        # Position of ``value`` within ``self`` in units of ``1/total``.
        pos = (value - self.lower) * total / self.width
        cumulative = 0
        for i, c in enumerate(counts):
            nxt = cumulative + c
            if pos < nxt or i == len(counts) - 1:
                return i
            cumulative = nxt
        return len(counts) - 1  # pragma: no cover - defensive

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = float(self.lower), float(self.upper)
        return f"Interval([{lo:.6f}, {hi:.6f}] = {self.num_lo}/{self.num_hi}/{self.den})"
