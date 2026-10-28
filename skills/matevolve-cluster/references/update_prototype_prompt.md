# Prototype Update Prompt — Step 4 (M-step, refresh the centroid)

Rewrite one cluster's prototype text from a sample of its current members, and
optionally flag it as heterogeneous (a qualitative split hint — the numeric split
decision is made later by the Python tool). Substitute `{resolution_rule}`, the
current `{label}`/`{prototype}`, and sampled members into `{members}`.

---

## SYSTEM PROMPT

You are refining the textual prototype (the "centroid") of one cluster of crystal
motifs, using geometry only.

{resolution_rule}

Given a sample of the cluster's members:
1. Write the **shared geometry** at Medium resolution — name the common unit
   class, dimensionality, coordination archetype, and CN band. Do not over-
   generalize to a vague catch-all, and do not narrow to one member's quirks.
2. Pick 2–3 `exemplars` (member ids) that best typify the cluster.
3. Set `heterogeneous = true` only if the sample clearly contains two or more
   distinct archetypes under the resolution rule; otherwise `false`.

## USER PROMPT

Current label: {label}
Current prototype: {prototype}

Sampled members (id and text):
{members}

Return STRICT JSON, no prose:

```json
{
  "label": "<=6-word archetype name",
  "prototype": "1-3 sentence geometry-only shared description naming all four axes",
  "exemplars": ["<member id>", "..."],
  "heterogeneous": true|false
}
```
