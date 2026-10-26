"""Geometry helpers (SI S1-S2), magnetization (Methods 6), prototype labels and rates (SI S4.1)."""

import math

from pymatgen.core import Lattice, Structure

from matevolve.geometry import ACCEPT_TOLERANCES, REPORT_TOLERANCES, bond_regime, geometry_report, structures_match, to_primitive, validate_cif
from matevolve.validation.magnetization import LANTHANIDE_4F, corrected_total_moment, magnetization
from matevolve.evaluation.prototype import prototype_label
from matevolve.evaluation.rates import Candidate, compute_rates


def test_bond_regimes_follow_covalent_radius_rule():
    # Fe-Fe: r_sum = 2.84 A (Cordero 1.42 x 2)
    assert bond_regime(2.48, "Fe", "Fe") == "[short bond]"        # 0.87 r_sum
    assert bond_regime(2.84, "Fe", "Fe") == "[typical bond]"
    assert bond_regime(3.40, "Fe", "Fe") == "[long bond]"         # 1.20 r_sum
    assert bond_regime(3.80, "Fe", "Fe") == "[non-bonded]"


def test_validate_cif_checks(smfe12):
    s, err = validate_cif(smfe12.to(fmt="cif"), "SmFe12")
    assert s is not None and err is None
    assert validate_cif("not a cif")[0] is None
    _, err = validate_cif(smfe12.to(fmt="cif"), "SmFe11Ti")
    assert "Composition mismatch" in err
    squeezed = Structure(Lattice.cubic(2.0), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])   # 1.73 A ... ok
    assert validate_cif(squeezed.to(fmt="cif"))[0] is not None
    overlapping = Structure(Lattice.cubic(4.0), ["Fe", "Fe"], [[0, 0, 0], [0.2, 0, 0]])     # 0.8 A
    _, err = validate_cif(overlapping.to(fmt="cif"))
    assert "Overlapping" in err and "1.5" in err


def test_geometry_report_lists_neighbours(bcc_fe):
    rep = geometry_report(bcc_fe)
    assert "Voronoi CN" in rep and "Fe-Fe" in rep and "[short bond]" in rep   # 2.49 A in bcc Fe


def test_structure_matcher_tolerances(smfe12):
    assert ACCEPT_TOLERANCES == dict(ltol=0.4, stol=0.7, angle_tol=15.0)
    assert REPORT_TOLERANCES == dict(ltol=0.3, stol=0.5, angle_tol=10.0)
    strained = smfe12.copy()
    strained.scale_lattice(smfe12.volume * 1.05)
    assert structures_match(smfe12, strained)
    prim = to_primitive(smfe12)
    assert len(prim) == 13 and prim.composition.reduced_formula == "SmFe12"


def test_4f_correction_signs_and_values(smfe12):
    gj_sm, sigma_sm = LANTHANIDE_4F["Sm"]
    assert math.isclose(gj_sm, 0.714, abs_tol=1e-3) and sigma_sm == +1
    assert LANTHANIDE_4F["Dy"][1] == -1 and 9.0 <= LANTHANIDE_4F["Dy"][0] <= 10.0
    # SmFe12 cell with 2 Sm: +2 x 0.714
    assert math.isclose(corrected_total_moment(smfe12.composition, 50.0), 50.0 + 2 * gj_sm)
    # Gd: 4f in valence, no correction
    assert math.isclose(corrected_total_moment({"Gd": 2, "Fe": 24}, 50.0), 50.0)
    m = magnetization(smfe12, 50.0)
    # mu0 Ms = mu0 |mu| mu_B / V : 51.43 mu_B in 343.9 A^3 -> ~1.74 T
    assert 1.5 < m.mu0Ms_T < 2.0


def test_prototype_label_is_anonymized(smfe12):
    lab = prototype_label(smfe12)
    dy = smfe12.copy()
    dy.replace_species({"Sm": "Dy"})
    assert prototype_label(dy) == lab                     # substitution keeps the label
    assert lab.startswith("AB12_tI26_139_")              # ThMn12: tI26, I4/mmm (139)
    bcc = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert prototype_label(bcc) != lab


def test_rates_sun_unique_novel(smfe12, bcc_fe):
    dup = smfe12.copy()
    dup.scale_lattice(smfe12.volume * 1.02)
    cands = [Candidate("a", smfe12, 0.05, 1.5), Candidate("b", dup, 0.08, 1.5),          # duplicate class -> one counted
             Candidate("c", bcc_fe, 0.0, 2.1), Candidate("d", bcc_fe.copy(), 0.3, 0.5)]    # c is known (in D_ref)
    reference = lambda formula: [bcc_fe] if formula == "Fe" else []
    rep = compute_rates(cands, reference, reference_labels={prototype_label(bcc_fe)})
    assert rep.unique == {"a", "c"} or rep.unique == {"a", "d"} or "a" in rep.unique
    assert "b" not in rep.unique
    assert rep.novel == {"a", "b"}
    assert rep.sun == {"a"}
    assert rep.magnetic == {"a"} and rep.new_prototype == {"a"} and rep.joint == {"a"}
    r = rep.rates()
    assert r["SUN"] == (1, 0.25) and r["joint"] == (1, 0.25)


def test_prototype_label_is_cell_invariant(smfe12):
    from matevolve.geometry import to_primitive
    lab = prototype_label(smfe12)
    assert prototype_label(to_primitive(smfe12)) == lab
    assert prototype_label(smfe12 * (2, 1, 1)) == lab
    assert lab == "AB12_tI26_139_a_fij"


def test_4f_correction_follows_sign_of_dft_moment(smfe12):
    gj = LANTHANIDE_4F["Sm"][0]
    assert math.isclose(abs(corrected_total_moment(smfe12.composition, -50.0)), 50.0 + 2 * gj)
