import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nll_sections import nearest_neighbour_km, section_metrics


def test_nearest_neighbour_km():
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.5]])
    assert nearest_neighbour_km(pts, chunk=2) == pytest.approx([5.0, 0.5, 0.5])


def test_section_metrics_thin_structure():
    rng = np.random.default_rng(0)
    along = rng.uniform(0, 10, 500)
    depth = 2 + 0.5 * along + rng.normal(0, 0.05, 500)   # faille plongeante fine
    m = section_metrics(along, depth)
    assert m["n"] == 500
    assert m["epaisseur_km"] == pytest.approx(0.05, abs=0.01)
    assert m["rapport_axes"] < 0.05
