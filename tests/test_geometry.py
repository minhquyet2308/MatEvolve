"""Geometry helpers: bond regimes, CIF validity checks, neighbour report, StructureMatcher tolerances."""

import math

from pymatgen.core import Lattice, Structure

from matevolve.geometry import ACCEPT_TOLERANCES, REPORT_TOLERANCES, bond_regime, geometry_report, structures_match, to_primitive, validate_cif


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
