# Merge Prompt — Step 6 (write the fused prototype)

The decision to merge is made by the Python tool (`decide`, via the confusion
rate φ). This worker only writes the **fused prototype text** for a merge group.
Substitute `{resolution_rule}`, the prototypes being merged into `{prototypes}`,
and a sample of their combined members into `{members}`.

---

## SYSTEM PROMPT

You are writing a single prototype that covers several clusters the system has
decided to merge (their motifs were voted interchangeably between them, so they
are one group). Use geometry only.

{resolution_rule}

Write one prototype describing the shared geometry of all the members below, at
Medium resolution (unit class, dimensionality, coordination archetype, CN band),
and pick 2–3 exemplars. Do not re-litigate the merge; just describe the union.

## USER PROMPT

Prototypes being merged:
{prototypes}

Sampled combined members (id and text):
{members}

Return STRICT JSON, no prose:

```json
{
  "label": "<=6-word archetype name",
  "prototype": "geometry-only shared description naming all four axes",
  "exemplars": ["<member id>", "..."]
}
```
