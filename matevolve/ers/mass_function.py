"""Dempster-Shafer mass functions and Dempster's rule of combination."""

from __future__ import annotations

from typing import Dict, FrozenSet, Hashable, Iterable, Mapping, Tuple

Hypothesis = FrozenSet[Hashable]


class MassFunction(dict):
    """A basic probability assignment over the subsets of a finite frame.

    Keys are ``frozenset`` hypotheses, values non-negative masses summing to
    one.  ``frame`` is the frame of discernment Omega; mass not assigned to
    a focal element is placed on Omega (total ignorance).
    """

    def __init__(self, frame: Iterable[Hashable], masses: Mapping[Iterable[Hashable], float] | None = None) -> None:
        super().__init__()
        self.frame: Hypothesis = frozenset(frame)
        total = 0.0
        for h, v in (masses or {}).items():
            h = frozenset(h)
            if not h <= self.frame or not h:
                raise ValueError(f"hypothesis {set(h)} is not a non-empty subset of the frame {set(self.frame)}")
            self[h] = self.get(h, 0.0) + float(v)
            total += float(v)
        if total > 1.0 + 1e-9:
            raise ValueError(f"masses sum to {total} > 1")
        self[self.frame] = self.get(self.frame, 0.0) + (1.0 - total)

    # ------------------------------------------------------------------
    def mass(self, h: Iterable[Hashable]) -> float:
        return self.get(frozenset(h), 0.0)

    def belief(self, h: Iterable[Hashable]) -> float:
        h = frozenset(h)
        return sum(v for k, v in self.items() if k <= h)

    def plausibility(self, h: Iterable[Hashable]) -> float:
        h = frozenset(h)
        return sum(v for k, v in self.items() if k & h)

    def conflict(self, other: "MassFunction") -> float:
        """K = sum of products of masses on disjoint hypotheses."""
        return sum(v1 * v2 for h1, v1 in self.items() for h2, v2 in other.items() if not (h1 & h2))

    def combine(self, other: "MassFunction") -> "MassFunction":
        """Dempster's rule: (m_a (+) m_b)(w) = 1/(1-K) * sum_{w_k ∩ w_h = w} m_a(w_k) m_b(w_h)."""
        if self.frame != other.frame:
            raise ValueError("mass functions must share the frame of discernment")
        acc: Dict[Hypothesis, float] = {}
        conflict = 0.0
        for h1, v1 in self.items():
            if v1 == 0.0:
                continue
            for h2, v2 in other.items():
                if v2 == 0.0:
                    continue
                inter = h1 & h2
                if inter:
                    acc[inter] = acc.get(inter, 0.0) + v1 * v2
                else:
                    conflict += v1 * v2
        # normalize by the surviving mass (= 1 - K in exact arithmetic).  Dividing by the
        # separately accumulated 1 - K instead lets rounding errors grow geometrically when
        # thousands of pieces of evidence are fused and m(Omega) has shrunk to ~1e-20.
        norm = sum(acc.values())
        if norm <= 1e-300:
            raise ZeroDivisionError("total conflict: Dempster's rule is undefined")
        out = MassFunction(self.frame)
        out.clear()
        for h, v in acc.items():
            out[h] = v / norm
        out.setdefault(self.frame, 0.0)
        return out

    def __matmul__(self, other: "MassFunction") -> "MassFunction":  # m1 @ m2
        return self.combine(other)


    def __repr__(self) -> str:  # pragma: no cover
        items = ", ".join(f"{{{','.join(map(str, sorted(h, key=str)))}}}: {v:.4f}" for h, v in sorted(self.items(), key=lambda kv: len(kv[0])))
        return f"MassFunction({items})"


def combine_all(masses: Iterable[MassFunction], frame: Iterable[Hashable]) -> MassFunction:
    """Fuse any number of mass functions (order-independent)."""
    out = MassFunction(frame)
    for m in masses:
        out = out.combine(m)
    return out


