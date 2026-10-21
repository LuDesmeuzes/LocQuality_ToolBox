"""
nll_gridding.py
===============
Quadrillage des localisations NLL : mesure, pour chaque run listé dans
gridding.runs et pour chaque run NLL local de paths.nll_runs, les artefacts
de la recherche par octree et les paliers de profondeur.

NLL place l'hypocentre au centre de la cellule d'octree la plus probable.
Connaissant LOCGRID et LOCSEARCH OCT du profil, on retrouve pour chaque
origine le niveau de subdivision L de sa cellule finale (à partir de la
profondeur, qui tombe exactement sur z0 + (m + ½)·dz0/2^L) et donc la taille
de cette cellule. Les événements à cellule finale grossière sont ceux qui
dessinent le quadrillage sur les cartes.

Indicateurs par run :
  - niveau d'octree et taille de la cellule finale (horizontale) ;
  - part des événements qui partagent exactement leur épicentre avec un
    autre (à 1 m près) ;
  - paliers de profondeur : nombre d'événements à ±plateau_window_km de
    chaque profondeur entière, comparé au voisinage (fenêtres décalées de
    ±0,2 à ±0,4 km) ; toit de la grille de recherche ;
  - événements au-dessus du relief (si gridding.topo est renseigné).

Sorties (dans <output_dir>/nll_gridding/) :
  - gridding.csv        : une ligne par origine NLL et par run
  - gridding_stats.csv  : indicateurs par run
  - plateaus.csv        : excès d'événements à chaque profondeur entière
  - cells.png           : répartition des tailles de cellule finale par run
  - depths.png          : histogramme fin des profondeurs (pas de 50 m)

Utilisation :
    python nll_gridding.py
    python nll_gridding.py --config un_autre_config.yaml
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from loc_common import (
    SERIES, hist_compare, load_config, log, nll_origins, output_dir, parse_args,
    load_bulletin, save, setup_style,
)

# Fenêtres de « voisinage » pour le fond des paliers (km, de part et d'autre)
PLATEAU_BACKGROUND_OFFSETS = (-0.4, -0.3, -0.2, 0.2, 0.3, 0.4)

# Tolérance (km) pour décider qu'une profondeur est un centre de cellule.
# Les profondeurs du bulletin tombent sur le réseau à 0,1 mm près ; 1 cm
# évite qu'une profondeur quelconque tombe par hasard sur un réseau fin.
LEVEL_TOLERANCE_KM = 1e-5


def octree_geometry(run_cfg):
    """
    Taille (km) des cellules initiales de l'octree à partir de LOCGRID
    (nx ny nz x0 y0 z0 dx dy dz) et de LOCSEARCH OCT (nx ny nz).
    """
    nx, ny, nz, x0, y0, z0, dx, dy, dz = run_cfg["locgrid"]
    ox, oy, oz = run_cfg["oct"]
    return {"cell_x0": (nx - 1) * dx / ox, "cell_y0": (ny - 1) * dy / oy,
            "cell_z0": (nz - 1) * dz / oz, "z0": z0}


def local_run_params(run_dir):
    """
    (nom, {locgrid, oct}) d'un run NLL local, lus dans son run.yaml : pour
    un run local, LOCGRID et LOCSEARCH sont connus, inutile de les recopier
    dans gridding.runs.
    """
    meta = yaml.safe_load((Path(run_dir) / "run.yaml").read_text())
    locgrid = [float(v) for v in meta["locgrid"].split()[:9]]
    search = meta["locsearch"].split()
    if search[0] != "OCT":
        raise ValueError(f"{run_dir} : LOCSEARCH {search[0]} n'est pas une recherche par octree")
    return meta["nom"], {"locgrid": locgrid, "oct": [int(v) for v in search[1:4]]}


def octree_level(depth, z0, cell_z0, tol=LEVEL_TOLERANCE_KM, min_cell_km=0.002):
    """
    Niveau de subdivision de la cellule finale déduit de la profondeur :
    plus petit L tel que depth = z0 + (m + ½)·cell_z0/2^L à tol près.
    Les réseaux de deux niveaux successifs sont disjoints, L est donc unique.
    Renvoie -1 si la profondeur ne tombe sur aucun réseau.
    """
    depth = np.asarray(depth, dtype=float)
    level = np.full(len(depth), -1)
    lmax = int(np.floor(np.log2(cell_z0 / min_cell_km)))
    for L in range(lmax, -1, -1):
        cell = cell_z0 / 2 ** L
        u = (depth - z0) / cell - 0.5
        level[np.abs(u - np.round(u)) * cell < tol] = L
    return level


def read_topo(path):
    """
    Grille topographique au format ASCII GMT 4 (celui de topo.asc pour
    LOCTOPO_SURFACE) : en-tête « x_min: … nx: … », puis une altitude (m)
    par ligne, du nord au sud et d'ouest en est.
    """
    with open(path) as f:
        head = [next(f) for _ in range(9)]
    text = " ".join(head)
    num = lambda key: float(re.search(rf"{key}:\s*(\S+)", text).group(1))
    grid = np.loadtxt(path, skiprows=9).reshape(int(num("ny")), int(num("nx")))
    return {"grid_m": grid, "x_min": num("x_min"), "y_max": num("y_max"),
            "x_inc": num("x_inc"), "y_inc": num("y_inc")}


def ground_altitude_km(topo, lat, lon):
    """Altitude du sol (km) au nœud de la grille le plus proche de chaque point."""
    g = topo["grid_m"]
    i = np.round((topo["y_max"] - np.asarray(lat)) / topo["y_inc"]).astype(int)
    j = np.round((np.asarray(lon) - topo["x_min"]) / topo["x_inc"]).astype(int)
    return g[np.clip(i, 0, g.shape[0] - 1), np.clip(j, 0, g.shape[1] - 1)] / 1000.0


def plateau_table(depth, z_top, window, z_max=10):
    """Événements à ±window de chaque profondeur entière, rapportés au voisinage."""
    depth = np.asarray(depth, dtype=float)
    count = lambda z: np.sum(np.abs(depth - z) < window)
    rows = []
    for z in range(int(np.ceil(z_top)), z_max + 1):
        background = np.mean([count(z + s) for s in PLATEAU_BACKGROUND_OFFSETS
                              if z + s >= z_top])
        rows.append({"profondeur_km": z, "n": count(z), "fond": background,
                     "exces": count(z) / max(background, 1.0)})
    return pd.DataFrame(rows)


def analyse_run(bulletin, author, run_cfg, topo, window):
    geo = octree_geometry(run_cfg)
    df = nll_origins(bulletin, author).copy()
    df["octree_level"] = octree_level(df["depth"], geo["z0"], geo["cell_z0"])
    found = df["octree_level"] >= 0
    df["cell_h_m"] = np.where(found, 1000 * geo["cell_x0"] / 2.0 ** df["octree_level"], np.nan)
    df["cell_z_m"] = np.where(found, 1000 * geo["cell_z0"] / 2.0 ** df["octree_level"], np.nan)

    # Épicentres strictement identiques (à ~1 m près)
    key = df[["latitude", "longitude"]].round(5)
    df["shared_epicentre"] = key.groupby(["latitude", "longitude"])["latitude"].transform("size").to_numpy() > 1

    df["at_roof"] = df["depth"] < geo["z0"] + window
    df["on_km_node"] = np.abs(df["depth"] - np.round(df["depth"])) < window
    if topo is not None:
        df["ground_km"] = ground_altitude_km(topo, df["latitude"], df["longitude"])
        df["in_air"] = df["depth"] < -df["ground_km"]

    plateaus = plateau_table(df["depth"], geo["z0"], window)
    stats = {
        "run": author,
        "n": len(df),
        "niveau_trouve_pct": 100 * found.mean(),
        "cellule_mediane_m": df["cell_h_m"].median(),
        "cellule_gt_100m_pct": 100 * (df["cell_h_m"] > 100).mean(),
        "epicentre_partage_pct": 100 * df["shared_epicentre"].mean(),
        "toit_n": int(df["at_roof"].sum()),
        "dans_air_n": int(df["in_air"].sum()) if "in_air" in df else np.nan,
        "erh_median_km": df["erh_km"].median(),
        "erz_median_km": df["depth_unc_km"].median(),
    }
    return df, plateaus, stats


def plot_cells(results, path):
    """Part des événements par taille de cellule finale, une barre par run."""
    sizes = sorted({round(s) for df in results.values() for s in df["cell_h_m"].dropna()})
    x = np.arange(len(sizes))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (author, df) in enumerate(results.items()):
        pct = df["cell_h_m"].round().value_counts(normalize=True).reindex(sizes, fill_value=0) * 100
        ax.bar(x + (i - (len(results) - 1) / 2) * width, pct.to_numpy(), width,
               color=SERIES[i], label=f"{author} (n={len(df)})")
    ax.set_xticks(x, [f"{s:.0f}" for s in sizes])
    ax.set_xlabel("Taille horizontale de la cellule finale de l'octree (m)")
    ax.set_ylabel("Part des événements (%)")
    ax.set_title("Cellule finale de l'octree — plus elle est grande, plus le quadrillage se voit")
    ax.legend(loc="upper left")
    save(fig, path)


def plot_depths(results, z_top, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    bins = np.arange(np.floor(z_top), 10.05, 0.05)
    hist_compare(ax, {a: df["depth"] for a, df in results.items()}, bins, "Profondeur (km)")
    ax.set_title("Profondeurs au pas de 50 m — les pics fins sont des paliers")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_gridding")

    gcfg = config.get("gridding") or {}
    runs = dict(gcfg.get("runs") or {})
    for run_dir in config.get("paths", {}).get("nll_runs") or []:
        name, params = local_run_params(run_dir)
        runs.setdefault(name, params)
    if not runs:
        raise ValueError("gridding.runs ou paths.nll_runs doit être renseigné dans config.yaml")
    window = gcfg.get("plateau_window_km", 0.05)
    topo_path = gcfg.get("topo")

    bulletin = load_bulletin(config)
    topo = None
    if topo_path:
        log(f"Lecture de la topographie {topo_path}")
        topo = read_topo(topo_path)

    results, all_plateaus, all_stats = {}, [], []
    for author, run_cfg in runs.items():
        log(f"Run {author} : OCT {run_cfg['oct']}")
        df, plateaus, stats = analyse_run(bulletin, author, run_cfg, topo, window)
        results[author] = df
        all_plateaus.append(plateaus.assign(run=author))
        all_stats.append(stats)

    stats = pd.DataFrame(all_stats).set_index("run")
    log(stats.round(2).to_string(), 1)
    plateaus = pd.concat(all_plateaus)
    marked = plateaus[plateaus["exces"] >= 2]
    log("Paliers (au moins 2 fois le voisinage) :")
    log(marked[["run", "profondeur_km", "n", "fond", "exces"]].round(1).to_string(index=False), 1)

    pd.concat(results.values(), ignore_index=True).to_csv(out / "gridding.csv", index=False)
    stats.to_csv(out / "gridding_stats.csv")
    plateaus.to_csv(out / "plateaus.csv", index=False)
    log(f"→ {out / 'gridding_stats.csv'}", 1)

    log("Figures")
    z_top = min(octree_geometry(c)["z0"] for c in runs.values())
    plot_cells(results, out / "cells.png")
    plot_depths(results, z_top, out / "depths.png")


if __name__ == "__main__":
    main()
