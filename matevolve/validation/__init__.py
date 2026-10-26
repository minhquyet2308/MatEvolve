"""Post-processing of first-principles results (Methods 6).

The DFT relaxations are run by a separate tool (SI S8 parameters).  This
package handles the hand-off (:mod:`dft_io`) and turns the returned total
energy and total moment into the validated quantities: energy above the
convex hull (:mod:`hull`) and 4f-corrected saturation magnetization
(:mod:`magnetization`).
"""

from .dft_io import DFTResult, export_candidates, read_dft_results, read_dft_subfolders, write_dft_subfolders, write_results_template
from .hull import STABILITY_EPSILON, MPHull, formation_energy_per_atom
from .magnetization import LANTHANIDE_4F, MagnetizationResult, corrected_total_moment, magnetization, saturation_magnetization_T

__all__ = ["DFTResult", "export_candidates", "read_dft_results", "read_dft_subfolders", "write_dft_subfolders", "write_results_template", "STABILITY_EPSILON", "MPHull",
           "formation_energy_per_atom", "LANTHANIDE_4F", "MagnetizationResult", "corrected_total_moment", "magnetization",
           "saturation_magnetization_T"]
