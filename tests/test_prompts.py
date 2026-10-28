"""Prompt files load and keep their JSON examples intact when filled."""

from pathlib import Path

from matevolve.prompts import PROMPT_ROOT, fill, load_prompt

REQUIRED = ["cif2text/system.txt", "cif2text/user.txt", "standardizer/system.txt", "standardizer/user.txt",
            "standardizer/motif_vocabulary.md", "text2cif/system.txt", "text2cif/user.txt",
            "evo_operator/system.txt", "evo_operator/crossover.txt", "evo_operator/mutation.txt",
            "coverage/system.txt", "coverage/user.txt"]
SKILL_FILES = ["SKILL.md", "scripts/cluster_tools.py", "references/resolution_rule.md", "references/assign_prompt.md",
               "references/seed_prompt.md", "references/update_prototype_prompt.md", "references/merge_prompt.md",
               "references/split_prompt.md", "references/consensus_rules.md"]


def test_all_prompt_files_present():
    for rel in REQUIRED:
        assert (PROMPT_ROOT / rel).is_file(), rel
        assert load_prompt(rel).strip()
    skill = Path(__file__).resolve().parents[1] / "skills" / "matevolve-cluster"
    for rel in SKILL_FILES:                      # the clustering / assignment skill ships with the repository
        assert (skill / rel).is_file(), rel
    skills = skill.parent                        # the framework skills call the same CLI with the agent transport
    assert (skills / "README.md").is_file() and (skills / "references" / "worker_prompt.md").is_file()
    for name, cmd in (("matevolve-structure-translate", "structure-translate"), ("matevolve-motif-translate", "motif-translate"),
                      ("matevolve-loop", "rank")):
        text = (skills / name / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\nname: " + name), name
        assert "run_matevolve.py" in text and cmd in text and "configs/agent_run.yaml" in text, name
    assert (skill.parents[1] / "configs" / "agent_run.yaml").is_file()


def test_fill_keeps_json_braces():
    t = load_prompt("text2cif/user.txt")
    out = fill(t, composition="SmFe12", text_description="motifs")
    assert "Composition: SmFe12" in out and "{composition}" not in out
    sys_prompt = load_prompt("evo_operator/system.txt")
    assert '"designs": [' in fill(sys_prompt)          # braces of the JSON example untouched


def test_si_placeholders():
    assert "{cif_content}" in load_prompt("cif2text/user.txt")
    assert "{composition}" in load_prompt("text2cif/user.txt") and "{text_description}" in load_prompt("text2cif/user.txt")
    assert "{operator}" in load_prompt("evo_operator/crossover.txt") and "{description_2}" in load_prompt("evo_operator/crossover.txt")
    assert "{description_2}" not in load_prompt("evo_operator/mutation.txt")


def test_extract_json_tolerates_trailing_comma_and_prose():
    from matevolve.llm import extract_json
    reply = 'Here is the design:\n{\n  "designs": [\n    {"motifs": [{"motif #1": "x"}], "composition": "SmFe12", "text_description": "d"}\n  ],\n}\n'
    obj = extract_json(reply)
    assert isinstance(obj, dict) and obj["designs"][0]["composition"] == "SmFe12"
    assert extract_json("```json\n[1, 2]\n```", expect=list) == [1, 2]
