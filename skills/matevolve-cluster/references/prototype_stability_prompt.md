# Prototype Stability Prompt — Step 6½ (the LLM "centroid-shift" test)

After the M-step rewrites prototypes, this worker compares each cluster's
**previous** prototype with its **updated** one and decides whether they still
describe the **same archetype** under the resolution rule. This is the
LLM-natural analogue of the k-means centroid-shift test: it tolerates pure
wording changes and only flags a real shift in geometry meaning. The Python tool
then tallies the yes/no verdicts into the prototype-stability fraction used by the
convergence test. Several clusters are compared in one call. Substitute
`{resolution_rule}` and the list of pairs into `{pairs}` (each: `cluster_id`,
`previous`, `updated`).

Clusters whose membership did not change since last round may be omitted (they are
stable by construction); only changed clusters need comparison.

---

## SYSTEM PROMPT

You are checking whether a cluster's prototype has materially changed between two
rounds of a clustering, using geometry only.

{resolution_rule}

For each pair, compare the `previous` and `updated` prototype descriptions and
decide:
- `stable = true` if they describe the **same archetype** under the resolution
  rule — i.e. the same unit class, dimensionality, coordination archetype, and CN
  band. Differences in wording, ordering, or added detail do NOT count as a
  change.
- `stable = false` if the updated prototype has shifted to a **different**
  archetype on any of those four axes.

Judge each pair independently. Do not score or explain.

## USER PROMPT

Pairs:
{pairs}

Return STRICT JSON, no prose:

```json
{ "results": [ {"cluster_id": <int>, "stable": true|false} ] }
```

Every pair must appear exactly once.
