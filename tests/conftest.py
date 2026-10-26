import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Lattice, Structure  # noqa: E402


@pytest.fixture
def smfe12() -> Structure:
    """ThMn12-type SmFe12 (I4/mmm), the host of the magnet run."""
    a, c = 8.50, 4.76
    lattice = Lattice.tetragonal(a, c)
    species, coords = [], []
    # Sm 2a; Fe 8f, 8i, 8j (conventional cell, 26 atoms)
    sm = [(0, 0, 0), (0.5, 0.5, 0.5)]
    f8 = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
          (0.75, 0.75, 0.75), (0.25, 0.25, 0.75), (0.25, 0.75, 0.25), (0.75, 0.25, 0.25)]
    x = 0.361
    i8 = [(x, 0, 0), (-x, 0, 0), (0, x, 0), (0, -x, 0), (x + .5, .5, .5), (-x + .5, .5, .5), (.5, x + .5, .5), (.5, -x + .5, .5)]
    y = 0.277
    j8 = [(y, .5, 0), (-y, .5, 0), (.5, y, 0), (.5, -y, 0), (y + .5, 0, .5), (-y + .5, 0, .5), (0, y + .5, .5), (0, -y + .5, .5)]
    for p in sm:
        species.append("Sm"); coords.append(p)
    for p in f8 + i8 + j8:
        species.append("Fe"); coords.append(p)
    return Structure(lattice, species, coords)


@pytest.fixture
def bcc_fe() -> Structure:
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
