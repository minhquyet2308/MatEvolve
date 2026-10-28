# Split Prompt — Step 6 (name the children of a split)

The decision to split is made by the Python tool (`decide`, via contested
fraction + spread leakage + a co-association cut) or by a `heterogeneous` flag
from the prototype-update worker. This worker only **names and describes** the
child prototypes. Substitute `{resolution_rule}`, the parent `{label}`/
`{prototype}`, and the candidate subgroups into `{subgroups}`.

`{subgroups}` is either:
- the Python co-association sub-groups (each a list of member id+text), or
- a single pooled member sample (when the cluster was flagged `heterogeneous`
  with no precomputed cut), in which case you propose the partition yourself.

---

## SYSTEM PROMPT

You are describing the sub-archetypes of a crystal-motif cluster that has been
judged internally inconsistent. Use geometry only.

{resolution_rule}

For each provided subgroup, write a child prototype (shared geometry at Medium
resolution: unit class, dimensionality, coordination archetype, CN band). Each
child must be a genuinely different group under the resolution rule — they must
differ on one of the four axes, NOT merely on shell composition, connectivity,
anisotropy, chemistry, or magnetism. If two provided subgroups are actually the
same archetype, return them as one child and list both group indices in
`from_groups`. If a single pooled sample was given, partition it yourself by
assigning each member id to a child.

## USER PROMPT

Parent label: {label}
Parent prototype: {prototype}

Candidate subgroups (group_index → members id+text):
{subgroups}

Return STRICT JSON, no prose:

```json
{
  "children": [
    {
      "label": "<=6-word archetype name",
      "prototype": "geometry-only shared description naming all four axes",
      "from_groups": [<group_index, ...>],
      "member_ids": ["<id>", "..."]
    }
  ]
}
```

Every member must appear in exactly one child.
