"""Algorithms 2 and 3 driven by a scripted model (no API calls)."""

import json

from pymatgen.core import Lattice, Structure

from matevolve.agents import CIF2Text, CoverageJudge, EvoOperator, Text2CIF
from matevolve.llm import ScriptedClient
from matevolve.profile import DesignConcept
from matevolve.translate import Translator


def cif2text_reply(n=2):
    return json.dumps({"motifs": [{"motif_id": str(i + 1), "name": f"m{i + 1}", "description": f"raw motif {i + 1}"} for i in range(n)],
                       "hard_magnet_assessment": {"verdict": "soft"}})


def standardizer_reply(n=2):
    return "\n".join(f"Motif #{i + 1}: A [network] of [framework] Fe in [cubic] coordination with [medium CN] 8, [pure shell] of 8 Fe, [3D] percolating." for i in range(n))


def coverage_reply(n_a, n_b):
    return json.dumps({"direction": "A->B", "items": [{"motif": i + 1, "covered_by": [min(i + 1, n_b)], "confidence": 0.95} for i in range(n_a)]})


def make_router(cif_text: str, n_motifs=2, fail_first_cif=False):
    """Answer each prompt by recognising which agent sent it."""
    state = {"cif_calls": 0}

    def respond(messages):
        system = messages[0]["content"]
        last = messages[-1]["content"]
        if system.startswith("You are an expert materials scientist specializing"):
            return cif2text_reply(n_motifs)
        if system.startswith("You are a crystal-structure motif standardizer"):
            return standardizer_reply(n_motifs)
        if system.startswith("You are a crystallographic structure generator"):
            state["cif_calls"] += 1
            if fail_first_cif and state["cif_calls"] == 1:
                return "data_bad\n_cell_length_a 1.0\nthis is not parseable"
            return cif_text
        if system.startswith("You are an expert crystallographer deciding"):
            n_a = last.split("## LIST A")[1].split("## LIST B")[0].strip().count("\n") + 1
            n_b = last.split("## LIST B")[1].split("Return JSON")[0].strip().count("\n") + 1
            return coverage_reply(n_a, n_b)
        if system.startswith("You are a materials design assistant"):
            return json.dumps({"designs": [{"motifs": [{"motif #1": "A [network] of [framework] Fe ..."}],
                                            "composition": "Fe", "text_description": "bcc iron framework"}]})
        raise AssertionError("unexpected prompt: " + system[:60])
    return respond, state


def build(bcc_fe, fail_first_cif=False):
    respond, state = make_router(bcc_fe.to(fmt="cif"), fail_first_cif=fail_first_cif)
    llm = ScriptedClient(responder=respond)
    translator = Translator(CIF2Text(llm, use_geometry_tool=True), Text2CIF(llm, prerelax=False), CoverageJudge(llm))
    return llm, translator, state


def test_structure_translate_accepts_round_trip(bcc_fe):
    llm, translator, _ = build(bcc_fe)
    res = translator.structure_translate(bcc_fe, source="bcc")
    assert res.accepted and len(res.attempts) == 1
    assert res.profile.composition == "Fe" and len(res.profile) == 2
    assert res.profile.motifs[0].description_standardized.startswith("A [network]")
    assert res.attempts[0].structure_match is True and res.attempts[0].coverage["passed"]
    # the profile carries no coordinates / lattice / space group
    txt = res.profile.text_description()
    assert "2.87" not in txt and "Im-3m" not in txt


def test_structure_translate_rejects_when_geometry_differs(smfe12):
    # same composition and cell, but Sm placed on two of the Fe 8f sites: coverage (scripted) passes, geometry does not
    species = ["Fe"] * 2 + ["Sm"] * 2 + ["Fe"] * 22
    wrong = Structure(smfe12.lattice, species, smfe12.frac_coords)
    respond, _ = make_router(wrong.to(fmt="cif"))        # Text2CIF always rebuilds the wrong arrangement
    llm = ScriptedClient(responder=respond)
    translator = Translator(CIF2Text(llm), Text2CIF(llm, prerelax=False), CoverageJudge(llm), translate_budget=2)
    res = translator.structure_translate(smfe12, source="smfe12")
    assert not res.accepted and len(res.attempts) == 2
    assert res.attempts[0].coverage["passed"] and res.attempts[0].structure_match is False
    assert "does not match" in res.attempts[0].feedback


def test_motif_translate_realizes_design_and_retries_validity(bcc_fe):
    llm, translator, state = build(bcc_fe, fail_first_cif=True)
    design = DesignConcept(composition="Fe", text_description="bcc iron", motifs=["A [network] of [framework] Fe ..."])
    res = translator.motif_translate(design)
    assert res.accepted and res.structure.composition.reduced_formula == "Fe"
    assert state["cif_calls"] == 2                      # one invalid CIF, one valid
    assert res.attempts[0].validity_issues and res.attempts[0].coverage["passed"]
    # the ##ISSUE## feedback went back into the Text2CIF conversation
    t2c_calls = [c for c in llm.calls if c[0]["content"].startswith("You are a crystallographic structure generator")]
    assert any("##ISSUE##" in m["content"] for m in t2c_calls[-1])


def test_evo_operator_parses_strict_json(bcc_fe):
    respond, _ = make_router(bcc_fe.to(fmt="cif"))
    llm = ScriptedClient(responder=respond)
    op = EvoOperator(llm)
    from matevolve.profile import Motif, MotifProfile
    p1 = MotifProfile("Fe", [Motif(1, description_standardized="A [network] ...")], source="p1")
    p2 = MotifProfile("SmFe12", [Motif(1, description_standardized="A [cage] ...")], source="p2")
    designs = op.propose([p1, p2], operator="CROSSOVER")
    assert len(designs) == 1 and designs[0].composition == "Fe" and designs[0].parents == ["p1", "p2"]
    user = llm.calls[-1][1]["content"]
    assert "## PARENT A" in user and "## PARENT B" in user and "CROSSOVER" in user
