"""End-to-end run of the agentic clustering with a scripted model."""

import json
import re

from matevolve.clustering import ClusterConfig, MotifCluster
from matevolve.llm import ScriptedClient

MOTIFS = {f"m{i}": ("A [cage] of [center] Sm ..." if i < 4 else "A [layer] of [framework] Fe ...") for i in range(8)}


def responder(messages):
    system, user = messages[0]["content"], messages[-1]["content"]
    if "propose a set of distinct" in system or "archetype prototypes" in system:
        return json.dumps({"prototypes": [{"label": "cage", "prototype": "cage, 3D, high CN", "exemplar_indices": [0]},
                                          {"label": "layer", "prototype": "layer, 2D, medium CN", "exemplar_indices": [4]}]})
    if "matching crystal motif descriptions to the prototype" in system:
        ids = re.findall(r"- id: (\S+)", user.split("Motifs to vote on:")[1])
        protos = dict(re.findall(r"cluster_id: (\d+)\n  label: (\w+)", user))
        by_label = {v: int(k) for k, v in protos.items()}
        return json.dumps({"votes": [{"id": i, "label": by_label["cage"] if "cage" in MOTIFS[i] else by_label["layer"]} for i in ids]})
    if "refining the textual prototype" in system:
        label = re.search(r"Current label: (\w+)", user).group(1)
        return json.dumps({"label": label, "prototype": f"{label} prototype", "exemplars": [], "heterogeneous": False})
    raise AssertionError(system[:80])


def test_cluster_converges_and_assignment_matches_prototypes(tmp_path):
    mc = MotifCluster(ScriptedClient(responder=responder), MOTIFS, tmp_path, ClusterConfig(batch_size=3, seed_sample=8))
    result = mc.run()
    assert result["K"] == 2 and not result["outliers"]
    members = {g["label"]: sorted(g["members"]) for g in result["groups"].values()}
    assert members == {"cage": ["m0", "m1", "m2", "m3"], "layer": ["m4", "m5", "m6", "m7"]}
    # stopped by the convergence rule (two rounds with s_bar >= 0.8 and ARI >= 0.95), not the cap
    assert len(result["history"]) == 3 and result["history"][-1]["ari_to_previous"] == 1.0
    assert (tmp_path / "clusters.json").is_file() and (tmp_path / "assignments.csv").is_file()
    # frozen assignment of a new motif
    frozen = MotifCluster.from_clusters(ScriptedClient(responder=responder), tmp_path / "clusters.json", tmp_path / "frozen",
                                        ClusterConfig(batch_size=3))
    MOTIFS["new1"] = "A [cage] of [center] Nd ..."
    assign = frozen.assign_frozen({"new1": MOTIFS["new1"]})
    cage_id = [int(k) for k, g in result["groups"].items() if g["label"] == "cage"][0]
    assert assign == {"new1": cage_id}
