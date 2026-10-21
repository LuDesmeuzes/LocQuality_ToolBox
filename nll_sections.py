"""
nll_sections.py
===============
Netteté des structures : cartes et coupes côte à côte pour les runs NLL de
compare.nll_authors, sur les mêmes événements, avec des mesures de la
concentration de la sismicité.

Une faille est fine ; les erreurs de localisation étalent la sismicité
autour. À événements égaux, le run qui dessine des structures plus fines et
plus resserrées localise probablement mieux. Attention : le quadrillage
(événements empilés sur un même point, voir nll_gridding.py) rend la
sismicité artificiellement « nette » ; la part de plus proches voisins
confondus (< 10 m) est donnée pour le repérer.

Mesures par run et par coupe (événements à ±half_width_km du profil) :
  - distance médiane au plus proche voisin dans le plan de la coupe (km) ;
  - épaisseur : écart-type selon le petit axe du nuage (ACP), et rapport
    petit axe / grand axe (faible = structure allongée) ;
  - part des plus proches voisins confondus (< 10 m).
Et sur toute la zone : distance médiane au plus proche voisin en 3D.

Deux sous-ensembles d'événements, comme nll_station_residuals.py : « tous »
(communs à tous les runs) et ceux classés dans
compare.well_constrained_classes par chacun des runs.

Sorties (dans <output_dir>/nll_sections/<sous-ensemble>/) :
  - metrics.csv   : une ligne par run et par coupe (+ « zone » pour la 3D)
  - maps.png      : une carte par run, avec les profils de coupe
  - sections.png  : une ligne par run, une colonne par coupe

Utilisation :
    python nll_sections.py
    python nll_sections.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    INK, INK_MUTED, SERIES, load_bulletin, load_config, lonlat_to_km, log, nll_origins,
    output_dir, parse_args, project_on_section, quality_subsets, require, save, setup_style,
)

# En deçà (km), deux événements sont considérés comme superposés
SAME_POINT_KM = 0.01


def nearest_neighbour_km(points, chunk=1000):
    """Distance de chaque point à son plus proche voisin (points : N × d, en km)."""
    points = np.asarray(points, dtype=float)
    out = np.full(len(points), np.nan)
    if len(points) < 2:
        return out
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        d = np.sqrt(((block[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
        d[np.arange(len(block)), np.arange(start, start + len(block))] = np.inf
        out[start:start + len(block)] = d.min(axis=1)
    return out


def section_metrics(along, depth):
    """Mesures de concentration d'un nuage (distance le long du profil, profondeur)."""
    pts = np.column_stack([along, depth])
    row = {"n": len(pts)}
    if len(pts) < 3:
        return row
    nn = nearest_neighbour_km(pts)
    eig = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))
    row.update(nn_median_km=np.median(nn), nn_confondus_pct=100 * np.mean(nn < SAME_POINT_KM),
               epaisseur_km=np.sqrt(eig[0]), longueur_km=np.sqrt(eig[1]),
               rapport_axes=np.sqrt(eig[0] / eig[1]))
    return row


def plot_maps(data, sections, zone, title, path):
    fig, axes = plt.subplots(1, len(data), figsize=(5 * len(data), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (author, df)) in enumerate(zip(axes, data.items())):
        ax.scatter(df["longitude"], df["latitude"], s=2, color=SERIES[i], linewidths=0)
        for sec in sections:
            ax.plot([sec["lon1"], sec["lon2"]], [sec["lat1"], sec["lat2"]], color=INK, linewidth=1)
            ax.annotate(sec.get("name", ""), (sec["lon1"], sec["lat1"]), fontsize=8,
                        color=INK, xytext=(3, 3), textcoords="offset points")
        if zone:
            ax.set_xlim(zone["lon_min"], zone["lon_max"])
            ax.set_ylim(zone["lat_min"], zone["lat_max"])
        ax.set_aspect(1 / np.cos(np.radians(df["latitude"].mean())))
        ax.set_xlabel("Longitude (°)")
        ax.set_title(f"{author} (n={len(df)})", fontsize=9)
    axes[0].set_ylabel("Latitude (°)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, path)


def plot_sections(data, sections, metrics, title, path):
    """Une ligne par run, une colonne par coupe ; mêmes axes pour tous les runs."""
    depths = np.concatenate([df["depth"].to_numpy() for df in data.values()])
    zmin, zmax = np.floor(np.nanmin(depths)), np.ceil(np.nanpercentile(depths, 99.5))
    fig, axes = plt.subplots(len(data), len(sections), squeeze=False,
                             figsize=(4.5 * len(sections), 2.6 * len(data) + 1),
                             sharey=True)
    for i, (author, df) in enumerate(data.items()):
        for j, sec in enumerate(sections):
            ax = axes[i, j]
            along, m = project_on_section(df["longitude"], df["latitude"], sec)
            ax.scatter(along[m], df["depth"].to_numpy()[m], s=2, color=SERIES[i], linewidths=0)
            mt = metrics[(metrics["run"] == author) & (metrics["coupe"] == sec.get("name"))].iloc[0]
            ax.set_title(f"{sec.get('name', '')} : n={mt['n']:.0f}, "
                         f"épaisseur {mt.get('epaisseur_km', np.nan):.2f} km, "
                         f"voisin {1000 * mt.get('nn_median_km', np.nan):.0f} m", fontsize=8)
            ax.set_ylim(zmax, zmin)
            ax.set_aspect("equal", adjustable="box")
            if j == 0:
                ax.set_ylabel(f"{author}\nProfondeur (km)", fontsize=8)
            if i == len(data) - 1:
                ax.set_xlabel("Distance le long du profil (km)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.99, 0.005, "épaisseur : écart-type selon le petit axe du nuage ; "
             "voisin : distance médiane au plus proche voisin dans la coupe",
             ha="right", fontsize=7, color=INK_MUTED)
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_sections")

    runs = list(require(config, "compare.nll_authors"))
    classes = require(config, "nll.quality_classes")
    good = config["compare"].get("well_constrained_classes") or ["A", "B"]
    sections = require(config, "map.sections")
    zone = config["map"].get("zone")

    bulletin = load_bulletin(config)
    origins = {r: nll_origins(bulletin, r) for r in runs}
    subsets = quality_subsets(origins, classes, good)
    lon0 = np.mean([s["lon1"] for s in sections])
    lat0 = np.mean([s["lat1"] for s in sections])

    for name, events in subsets.items():
        sub_out = out / name
        sub_out.mkdir(exist_ok=True)
        data = {r: df[df["event_id"].isin(events)] for r, df in origins.items()}
        log(f"Sous-ensemble {name} : {len(events)} événements")
        rows = []
        for author, df in data.items():
            x, y = lonlat_to_km(df["longitude"], df["latitude"], lon0, lat0)
            nn3 = nearest_neighbour_km(np.column_stack([x, y, df["depth"]]))
            rows.append({"run": author, "coupe": "zone", "n": len(df),
                         "nn_median_km": np.nanmedian(nn3),
                         "nn_confondus_pct": 100 * np.mean(nn3 < SAME_POINT_KM)})
            for sec in sections:
                along, m = project_on_section(df["longitude"], df["latitude"], sec)
                rows.append({"run": author, "coupe": sec.get("name"),
                             **section_metrics(along[m], df["depth"].to_numpy()[m])})
        metrics = pd.DataFrame(rows)
        log(metrics.round(3).to_string(index=False), 1)
        metrics.to_csv(sub_out / "metrics.csv", index=False)
        label = f"{name} — {len(events)} événements communs"
        plot_maps(data, sections, zone, f"Cartes ({label})", sub_out / "maps.png")
        plot_sections(data, sections, metrics, f"Coupes ({label})", sub_out / "sections.png")


if __name__ == "__main__":
    main()
