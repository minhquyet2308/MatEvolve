"""Orbital-field matrix (OFM) descriptor (Pham et al. 2017, 2020).

Encodes the coordination of every atom -- the valence-orbital types of the
atom and of its neighbours, weighted by the solid angles of its Voronoi
polyhedron and divided by the distance -- and averages the local matrices
into one fixed-length vector per structure (OFM1: 32 x 33 = 1056 entries).
Because the descriptor is an average over sites it needs no atom
correspondence; the distance between two structures is the L1 norm of the
difference of their vectors (SI S4.2).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict

import numpy as np
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Element, Structure

SUBSHELLS = ["s1", "s2", "p1", "p2", "p3", "p4", "p5", "p6",
             "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
             "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14"]


@lru_cache(maxsize=None)
def element_vector(symbol: str) -> np.ndarray:
    """One-hot valence-subshell representation of an element (32-dim)."""
    vec = {k: 0.0 for k in SUBSHELLS}
    if symbol == "H":
        shells = ["s1"]
    elif symbol == "He":
        shells = ["s2"]
    else:
        es = Element(symbol).electronic_structure
        shells = ["".join(p) for p in re.findall(r"\.\d(\w+)(\d+)", es)]
    for s in shells:
        if s in vec:
            vec[s] = 1.0
    return np.array([vec[k] for k in SUBSHELLS])


def ofm(structure: Structure, include_distance: bool = True, ofm1: bool = True) -> np.ndarray:
    """Structure-averaged OFM vector (1056-dim for OFM1)."""
    vnn = VoronoiNN(cutoff=30.0, allow_pathological=True)
    locals_: list = []
    for i, site in enumerate(structure):
        center = element_vector(site.specie.symbol)
        env = np.zeros(len(SUBSHELLS))
        for nn in vnn.get_nn_info(structure, i):
            d = float(np.linalg.norm(site.coords - nn["site"].coords))
            w = nn["weight"] / d if include_distance else nn["weight"]
            env += element_vector(nn["site"].specie.symbol) * w
        if ofm1:
            env = np.concatenate(([1.0], env))
        locals_.append(np.ravel(center[None, :] * env[:, None]))
    return np.mean(np.array(locals_), axis=0)


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())
