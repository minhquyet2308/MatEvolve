# Cross-Fold Match Prompt — consensus stage

After discovery has converged independently on each fold, this worker decides
which archetypes are the **same** across folds (so we can tell which prototypes are
sample-independent). Substitute `{resolution_rule}` and the collated fold
dictionaries into `{fold_dictionaries}` (the JSON `{fold: [{cluster_id, label,
prototype}]}` from `collate-folds`).

---

## SYSTEM PROMPT

You are comparing the geometry-archetype dictionaries discovered independently on
several disjoint samples ("folds") of the same corpus. Group archetypes that are
the **same** under the resolution rule, regardless of fold or cluster id or exact
wording.

{resolution_rule}

Rules:
- One group = one archetype, listing every (fold, cluster_id) that expresses it.
- Two prototypes are the same archetype when they match on unit class,
  dimensionality, coordination archetype, and CN band — differences in wording,
  shell composition, connectivity, or anisotropy do NOT make them different.
- Each (fold, cluster_id) belongs to exactly one group.
- Give each group a short canonical `archetype` label.

## USER PROMPT

Fold dictionaries:
{fold_dictionaries}

Return STRICT JSON, no prose:

```json
{
  "groups": [
    {
      "archetype": "<=6-word canonical archetype name",
      "members": [ {"fold": "fold_00", "cluster_id": 0}, {"fold": "fold_01", "cluster_id": 3} ]
    }
  ]
}
```

Every (fold, cluster_id) pair from the input must appear in exactly one group.
The orchestrator saves this as `consensus/fold_match.json`; `consensus` then tallies
core (in all folds) / majority (≥ half) / singleton (one fold).
