# Motif Vocabulary — geometry-only controlled terms

The vocabulary used by `matevolve-characterize` to standardize 3D-geometry-motif
descriptions of inorganic crystal structures. All terms are about **shape,
coordination, connectivity, and periodicity**. No property terms (exchange,
anisotropy, magnetic order, moment, etc.) appear here — those belong to
`matevolve-design`.

Standardized text uses square-bracketed vocabulary throughout.

---

## 1. MOTIF TYPE — exactly one per gene

| Value | Meaning |
|---|---|
| `polyhedron` | A central atom with a well-defined coordination shell (tetrahedron, octahedron, cuboctahedron, …) |
| `cage` | A central atom fully enclosed by a high-coordination shell (typically CN ≥ 9), often Frank-Kasper or icosahedral |
| `cluster` | A small grouping of atoms with internal structure, not centered on a single atom |
| `network` | Connected framework of bonded atoms extending across the cell |
| `layer` | A two-dimensional sheet of atoms within the unit cell |
| `chain` | A one-dimensional sequence of bonded atoms |
| `ring` | A closed cyclic arrangement of atoms |
| `dimer` | Two atoms bonded at a distinctively short distance |
| `isolated_site` | A single atom with no neighbours within typical bonding range |
| `void` | A periodic empty region of structural significance (e.g. zeolite cage, vacancy) |
| `sublattice` | The spatial arrangement of one species across the cell, independent of the others |
| `interpenetrating` | Two or more disjoint frameworks occupying the same volume |

---

## 2. ELEMENT GEOMETRIC ROLE

Roles describe **where** an atom sits in the motif, not what it does physically.

| Value | Meaning |
|---|---|
| `center` | The atom around which a polyhedron / cage is built |
| `vertex` | An atom at the corner of a polyhedron / cluster |
| `edge` | An atom along the edge of an extended motif |
| `face` | An atom on the face of a polyhedron / cage |
| `bridge` | An atom shared between two adjacent motifs (corner / edge / face linkage) |
| `terminal` | An atom that closes off an otherwise-extended motif |
| `spacer` | An atom that separates motifs without participating in their primary coordination |
| `interstitial` | An atom in a small site between the main framework atoms |
| `framework` | An atom of an extended (1D/2D/3D) backbone |

---

## 3. COORDINATION GEOMETRY

| Family | Members |
|---|---|
| Low CN | `linear`, `bent`, `trigonal-planar`, `trigonal-pyramidal`, `tetrahedral`, `square-planar` |
| Medium CN | `square-pyramidal`, `trigonal-bipyramidal`, `octahedral`, `pentagonal-bipyramidal`, `square-antiprismatic`, `tri-capped-trigonal-prismatic` |
| High CN | `cuboctahedral`, `icosahedral`, `Frank-Kasper-Z14`, `Frank-Kasper-Z15`, `Frank-Kasper-Z16` |
| Distortion qualifier | Prefix with `distorted-` (e.g. `distorted-octahedral`) when the polyhedron deviates noticeably from the ideal |
| Capping qualifier | Prefix with `mono-capped-`, `bi-capped-`, `tri-capped-` for capped polyhedra |

---

## 4. COORDINATION NUMBER BUCKET

| Tag | Range | Note |
|---|---|---|
| `[low CN]` | 2–4 | Append exact CN when known: `[low CN] 4` |
| `[medium CN]` | 5–8 | |
| `[high CN]` | 9–16 | |
| `[very high CN]` | > 16 | |

---

## 5. SHELL COMPOSITION

| Tag | Meaning |
|---|---|
| `[pure shell]` of `<Element>` | Coordination shell contains a single element |
| `[binary shell]` of `<A>` and `<B>` | Shell contains exactly two elements (with counts when known: `[binary shell] of 4 O + 2 F`) |
| `[mixed shell]` of `<A>`, `<B>`, `<C>` | Shell contains three or more elements |

---

## 6. BOND-DISTANCE REGIME (element-pair agnostic)

Distances are converted to regimes by comparing the observed distance `d` to the
sum of covalent radii `r_sum = r_cov(A) + r_cov(B)` of the bonded pair. This
removes the magnetic-element-pair specificity of the older topic tree.

| Tag | Condition | Meaning |
|---|---|---|
| `[short bond]` | `d < 0.90 · r_sum` | Compressed pair, strong covalent / metallic overlap |
| `[typical bond]` | `0.90 · r_sum ≤ d ≤ 1.10 · r_sum` | Standard bonding range for this pair |
| `[long bond]` | `1.10 · r_sum < d ≤ 1.30 · r_sum` | Stretched, weakly bonded |
| `[non-bonded]` | `d > 1.30 · r_sum` | No direct bond; pair belongs to different motifs |

When multiple pairs matter, list the pair label first: `[A–B] in [short bond] regime`.

For canonical covalent-radius values see Cordero et al. (2008), Dalton Trans., 2832; pymatgen's `pymatgen.core.periodic_table.Element.atomic_radius` may be used as a deterministic substitute.

---

## 7. CONNECTIVITY / TOPOLOGY (how motifs join)

| Tag | Meaning |
|---|---|
| `corner-sharing` | Adjacent polyhedra share a single vertex |
| `edge-sharing` | Adjacent polyhedra share two vertices (an edge) |
| `face-sharing` | Adjacent polyhedra share three or more vertices (a face) |
| `vertex-sharing` | Generic single-atom linkage between motifs |
| `isolated` | Motif has no connection to others |
| `percolating` | Connections extend across the entire cell |
| `disconnected` | Connections terminate within the cell |
| `chained` | Connections form a 1D pathway |
| `layered` | Connections form a 2D pathway |
| `framework` | Connections form a 3D pathway |
| `interpenetrating` | Two or more frameworks coexist without sharing atoms |

---

## 8. DIMENSIONALITY (extent of the motif's footprint)

`[0D]`, `[1D]`, `[2D]`, `[3D]`, `[quasi-1D]`, `[quasi-2D]`

Use quasi-`n`D when the motif is `n`-dimensional in primary coordination but
extends weakly into `n+1` via secondary linkages.

---

## 9. SPATIAL RELATIONSHIP TAGS (optional, for description sentence 2)

`in-plane`, `axial`, `equatorial`, `above`, `below`, `across-layer`, `inter-layer`, `intra-chain`, `inter-chain`, `body-diagonal`, `face-diagonal`, `at-origin`, `at-cell-center`.

---

## 10. STACKING / REPETITION PATTERN (optional)

`single-layer-repeat`, `alternating-layers`, `multi-layer-repeat`, `polytype-2H`, `polytype-3R`, `polytype-4H`, `polytype-6H`, `superstructure`.

---

## 11. SITE SYMMETRY

Standard crystallographic point groups, written as the Hermann–Mauguin symbol
in brackets: `[1]`, `[-1]`, `[2]`, `[m]`, `[2/m]`, `[222]`, `[mmm]`, `[4]`, `[-4]`, `[4/m]`, `[422]`, `[4mm]`, `[-42m]`, `[4/mmm]`, `[3]`, `[-3]`, `[32]`, `[3m]`, `[-3m]`, `[6]`, `[-6]`, `[6/m]`, `[622]`, `[6mm]`, `[-6m2]`, `[6/mmm]`, `[23]`, `[m-3]`, `[432]`, `[-43m]`, `[m-3m]`.

When the site point group is unknown, use a qualitative tag: `[centrosymmetric]`, `[axial]`, `[planar]`, `[chiral]`, `[isotropic]`.

---

## What is deliberately NOT in this vocabulary

- No exchange mechanism (`direct d-d`, `superexchange`, `RKKY`, `f-d`, etc.)
- No magnetic order (`ferromagnetic`, `antiferromagnetic`, `ferrimagnetic`)
- No moment type (`itinerant`, `localized`, `quenched`, `induced`)
- No anisotropy or CEF terms (`uniaxial CEF`, `easy axis`, etc.)
- No electronic-structure qualifiers (`p-d hybridization`, `band broadening`, `Stoner criterion`)
- No element classes by magnetic role (`magnetic TM`, `heavy RE strong spin-orbit`, `light metalloid`)

All of the above belong in `matevolve-design`, which reasons about *what the geometry implies for properties*. `matevolve-characterize` only reports *what is geometrically there*.
