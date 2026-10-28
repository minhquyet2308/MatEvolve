# Resolution Rule — the single definition of "same geometry group"

This block is included verbatim in **every** worker prompt. It is the only
thing that fixes how coarse or fine the clusters are. Do not improvise a
different resolution.

---

## Goal

Group motif descriptions so that members of one group share **similar geometric
characteristics** — the shape and spatial arrangement of the atoms. Ignore all
magnetic, exchange, electronic, and property content when judging similarity.
(The paragraphs still contain that text; you simply do not use it for grouping.)

## Resolution: MEDIUM / archetype level

Two motifs are the **same group** when, and only when, they match on all four
of these axes:

1. **Unit class** — isolated unit / cage / chain / layer / framework.
2. **Dimensionality** — 0D, quasi-1D, 1D, quasi-2D, 2D, 3D.
3. **Coordination archetype** — the named local shape, e.g. tetrahedral,
   octahedral, prismatic, square-net / square-planar, Frank-Kasper cage,
   icosahedral cage, RE cage, kagome layer, honeycomb layer, dimer / dumbbell,
   chain, ring, triangular/trigonal.
4. **Coordination-number band** — low / medium / high / very-high.

## Explicitly NOT grounds for a separate group (treat as the same group)

- Shell composition (pure-TM vs pure-X vs mixed TM+X vs mixed TM+RE).
- Connectivity / polyhedral sharing (corner- vs edge-sharing; percolating vs
  isolated vs connected).
- Bond anisotropy / layout (in-plane vs axial vs across-layer, layer repeat).
- Any element identity, chemistry, or magnetic/exchange/electronic statement.

Two motifs that differ **only** on the items in this list belong to the **same**
group.

## Quick test

> "Stripping away chemistry and magnetism, do these motifs have the same kind of
> structural unit, of the same dimensionality, built from the same coordination
> shape and a comparable coordination number?" If yes → same group.

## Worked examples

- *Fe4-tetrahedron boride unit, low CN, 3D network* and *Fe4-tetrahedron unit,
  low CN, 3D, mixed shell* → **same group** (shell differs only).
- *Frank-Kasper cage, high CN, 3D* and *icosahedral cage, very-high CN, 3D* →
  **same group only if** archetype+CN band are judged equivalent; if one is
  clearly "cage/high" and the other "cage/very-high", they are **different**
  groups (CN band differs).
- *kagome layer, quasi-2D, medium CN* and *honeycomb layer, quasi-2D, medium
  CN* → **different** groups (archetype differs: kagome ≠ honeycomb).
- *octahedral framework, 3D, medium CN, corner-sharing* and *octahedral
  framework, 3D, medium CN, edge-sharing* → **same group** (sharing differs
  only).
