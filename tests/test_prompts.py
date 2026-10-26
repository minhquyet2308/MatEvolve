"""Prompt files load and keep their JSON examples intact when filled."""

from pathlib import Path

from matevolve.prompts import PROMPT_ROOT, fill, load_prompt

REQUIRED = ["cif2text/system.txt", "cif2text/user.txt", "standardizer/system.txt", "standardizer/user.txt",
            "standardizer/motif_vocabulary.md", "text2cif/system.txt", "text2cif/user.txt",
            "evo_operator/system.txt", "evo_operator/crossover.txt", "evo_operator/mutation.txt",
            "coverage/system.txt", "coverage/user.txt", "cluster/resolution_rule.md",
            "cluster/assign_system.txt", "cluster/assign_user.txt", "cluster/seed_system.txt", "cluster/seed_user.txt",
            "cluster/update_prototype_system.txt", "cluster/update_prototype_user.txt",
            "cluster/merge_system.txt", "cluster/merge_user.txt", "cluster/split_system.txt", "cluster/split_user.txt"]


def test_all_prompt_files_present():
    for rel in REQUIRED:
        assert (PROMPT_ROOT / rel).is_file(), rel
        assert load_prompt(rel).strip()


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
