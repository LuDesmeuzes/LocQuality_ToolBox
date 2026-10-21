import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nll_gridding import octree_geometry, octree_level, plateau_table


def test_octree_geometry():
    geo = octree_geometry({"locgrid": [101, 101, 30, -50, -50, -5.0, 1.0, 1.0, 1.0],
                           "oct": [15, 15, 15]})
    assert np.isclose(geo["cell_x0"], 100 / 15)
    assert np.isclose(geo["cell_z0"], 29 / 15)
    assert geo["z0"] == -5.0


def test_octree_level_retrouve_le_niveau():
    cell_z0 = 29 / 15
    # Centre de cellule au niveau 6 (exemple du bulletin 1D_ubaye_v1)
    depth_l6 = -5 + (248 + 0.5) * cell_z0 / 2 ** 6
    depth_l2 = -5 + (3 + 0.5) * cell_z0 / 2 ** 2
    levels = octree_level([depth_l6, depth_l2, 1.2345678], -5.0, cell_z0)
    assert levels[0] == 6
    assert levels[1] == 2
    assert levels[2] == -1


def test_plateau_table_detecte_un_empilement():
    rng = np.random.default_rng(0)
    depth = np.concatenate([rng.uniform(-5, 10, 3000), np.full(50, -2.0)])
    t = plateau_table(depth, -5.0, 0.05).set_index("profondeur_km")
    assert t.loc[-2, "exces"] > 3
    assert t.loc[4, "exces"] < 2
