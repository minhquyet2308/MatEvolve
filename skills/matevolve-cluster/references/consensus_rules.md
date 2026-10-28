# Consensus Metrics & Decision Rules

Every metric here is computed by `scripts/cluster_tools.py` from the vote logs.
LLM workers never see or produce these numbers — they only cast categorical votes.
This file is the authoritative definition (cite it in a manuscript).

## Foundation — the vote tally
Each motif `m_i` (i = 1..N) is assigned `R` times. Let `a_r(i) ∈ {1,…,K, ∅}` be
the cluster label cast in replicate `r` (∅ = "none/fits nothing"). The vote count
is `n_{i,k} = #{ r : a_r(i) = k }`, with `Σ_k n_{i,k} + n_{i,∅} = R`.

## 1. Assignment & fit quality
- **Dominant assignment:** `d(i) = argmax_k n_{i,k}`.
- **Consensus:** `s_i = n_{i,d(i)} / R ∈ (0,1]`.
- **Confident** if `s_i ≥ s_star` (default 0.6); **contested** otherwise.
- **Outlier** if `d(i) = ∅` → routed to the birth pool.
- **Cluster purity:** `purity(k) = mean_{i∈C_k} s_i`, where `C_k = {i : d(i)=k}`.

"Poor fit" ≡ low consensus. No axis counting.

## 2. Merge rule — clusters that share their motifs
- **Confusion count:** `confuse(A,B) = #{ i : n_{i,A} > 0 AND n_{i,B} > 0 }`
  (motifs that received at least one vote for each of A and B).
- **Confusion rate:** `φ(A,B) = confuse(A,B) / min(|C_A|, |C_B|)`.
- **Merge** A and B when `φ(A,B) ≥ phi_merge` (default 0.5). Merge pairs are
  combined transitively (union-find) into merge groups.
- *Note:* `φ` can exceed 1 (a motif may be counted once toward a pair even though
  confuse can approach the smaller size); it is used only as a threshold signal.

## 3. Split rule — clusters whose members vote inconsistently
- **Contested fraction:** `q_A = #{ i∈C_A : s_i < s_star } / |C_A|`. Only clusters
  with `q_A ≥ q_split` (default 0.3) are considered for splitting.
- **Leakage:** for each contested `i∈C_A`, its runner-up cluster is the most
  common non-`A`, non-∅ label among its votes. `leak(A→B)` counts them.
  Concentration `= max_B leak(A→B) / #contested(A)`.
  - If concentration `≥ leak_conc` (default 0.6) → the instability points at one
    rival → treat as a **MERGE** of A with that rival (not a split).
  - Otherwise → candidate **SPLIT**.
- **Co-association cut:** for candidate splits, `C_ij = (1/R) #{ r : a_r(i)=a_r(j) ≠ ∅ }`
  over the cluster's members. Build a graph with an edge when `C_ij ≥ co_thresh`
  (default 0.5); its connected components of size `≥ max(4, 0.15|C_A|)` are the
  sub-groups. If ≥ 2 such components exist, **split** the cluster along them.
- A cluster the prototype-update worker flags `heterogeneous=true` is also split,
  even without a co-association cut (the split worker then partitions it).

## 4. Convergence
- **Mean consensus:** `s̄_t = (1/N) Σ_i s_i` (quality gate).
- **Partition stability:** `ρ_t = 1 − ARI(d_t, d_{t-1})`, where ARI is the
  Adjusted Rand Index between consecutive rounds' dominant assignments
  (label-invariant: relabeling from merge/split does not affect it).
  ARI from the contingency table `n_{uv} = |C_u^{(t)} ∩ C_v^{(t-1)}|`,
  row/col sums `α_u, β_v`, and `C(n,2)=n(n-1)/2`:

  ```
  ARI = ( Σ_uv C(n_uv,2) − [Σ_u C(α_u,2)·Σ_v C(β_v,2)] / C(N,2) )
        / ( ½[Σ_u C(α_u,2)+Σ_v C(β_v,2)] − [Σ_u C(α_u,2)·Σ_v C(β_v,2)] / C(N,2) )
  ```
- **Prototype stability** (the k-means "centroid shift → 0" analogue). Because the
  prototype is re-written by a *stochastic* LLM each M-step, `ρ_t` alone does not
  guarantee the centroids settled. The stability signal is **semantic and
  LLM-judged**, not a hard membership threshold: after the M-step, an LLM compares
  each cluster's previous vs updated prototype and returns `stable_k ∈ {0,1}` —
  whether they still describe the **same archetype** under the resolution rule
  (wording changes do not count). The round's **prototype-stability fraction** is
  `Σ_t = (1/K) Σ_k stable_k`. A round is prototype-stable when no structural change
  occurred and `Σ_t ≥ proto_stable_frac` (default 1.0 = all clusters; lower it for
  a looser criterion).
  - *Efficiency fast-path:* a cluster whose membership is literally unchanged
    (Jaccard `J_k = |C_k^{(t)} ∩ C_k^{(t-1)}| / |C_k^{(t)} ∪ C_k^{(t-1)}| ≥ j_freeze`,
    default 0.98) is taken as `stable_k = 1` without an LLM call, and need not be
    re-synthesized. If no LLM verdicts are supplied at all (e.g. a dry run), the
    tool falls back to the pure-Jaccard rule `min_k J_k ≥ j_freeze`.
- **Stop** when, for two consecutive rounds, `s̄_t ≥ s_min` (default 0.8) AND
  `ρ_t < eps` (default 0.05) AND the round is prototype-stable (no merge/split,
  `Σ_t ≥ proto_stable_frac`); or at `max_rounds`.

## Default thresholds (set at `init`, overridable)
| symbol | param | default | used by |
|---|---|---|---|
| `s_star` | s_star | 0.6 | consensus floor (fit, contested) |
| `phi_merge` | phi_merge | 0.5 | merge |
| `q_split` | q_split | 0.3 | split trigger |
| `leak_conc` | leak_conc | 0.6 | split-vs-merge |
| `co_thresh` | co_thresh | 0.5 | co-association cut |
| `s_min` | s_min | 0.8 | convergence quality gate |
| `eps` | eps | 0.05 | convergence stability gate |
| `proto_stable_frac` | proto_stable_frac | 1.0 | fraction of clusters that must be semantically stable to converge |
| `j_freeze` | j_freeze | 0.98 | membership fast-path / dry-run stability fallback |
| `R` | R | 5 | ensemble votes per motif |
| `outlier_floor` | outlier_floor | 15 | birth trigger |
| `max_rounds` | max_rounds | 8 | hard stop |
