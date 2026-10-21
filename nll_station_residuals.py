"""
nll_station_residuals.py
========================
Résidus par station des runs NLL locaux (paths.nll_runs) : pour chaque
station et chaque type de phase (P, S), le résidu moyen (biais : le modèle
de vitesse prédit systématiquement trop tôt ou trop tard pour cette
station) et sa dispersion, comparés entre les runs de compare.nll_authors.

Résidu = temps observé (pointé) − temps calculé par NLL depuis l'hypocentre.
Un bon modèle donne des biais proches de 0 à toutes les stations. Une partie
de l'erreur du modèle est absorbée par la position de l'hypocentre : les
résidus la sous-estiment.

Les runs sont comparés sur les mêmes événements, pour deux sous-ensembles :
« tous » (événements communs à tous les runs) et les événements classés dans
compare.well_constrained_classes (ex. A et B, selon nll.quality_classes)
par chacun des runs. Seules les phases utilisées (poids > 0) comptent.

Sorties (dans <output_dir>/nll_station_residuals/<sous-ensemble>/) :
  - stations.csv        : une ligne par run, station et type de phase
                          (n, moyenne, médiane, écart-type, distance médiane)
  - summary.csv         : une ligne par run (RMS des résidus, biais médian…)
  - station_bias.png    : biais moyen par station, un point par run
  - residual_vs_distance.png : résidu médian par tranche de distance

Utilisation :
    python nll_station_residuals.py
    python nll_station_residuals.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    INK_MUTED, SERIES, load_bulletin, load_config, load_nll_phases, log, nll_origins,
    output_dir, parse_args, quality_subsets, require, save, setup_style,
)

DISTANCE_BINS_KM = [0, 5, 10, 15, 20, 30, 40, 60, 80, 120, 200]


def station_stats(phases):
    """Statistiques des résidus par run, station et type de phase."""
    g = phases.groupby(["author", "station", "phase_type"])["residual_s"]
    stats = g.agg(n="size", moyenne_s="mean", mediane_s="median", ecart_type_s="std")
    stats["distance_mediane_km"] = phases.groupby(
        ["author", "station", "phase_type"])["dist_km"].median()
    return stats.reset_index()


def run_summary(phases, stats, min_count):
    """Indicateurs par run : RMS des résidus et biais des stations bien échantillonnées."""
    rows = []
    for author, ph in phases.groupby("author", sort=False):
        st = stats[(stats["author"] == author) & (stats["n"] >= min_count)]
        row = {"run": author, "phases": len(ph),
               "rms_residus_s": np.sqrt(np.mean(ph["residual_s"] ** 2)),
               "pct_residus_gt_0.5s": 100 * np.mean(ph["residual_s"].abs() > 0.5)}
        for t in ("P", "S"):
            s = st[st["phase_type"] == t]
            row[f"biais_median_abs_{t}_s"] = s["moyenne_s"].abs().median()
            row[f"ecart_type_median_{t}_s"] = s["ecart_type_s"].median()
            row[f"stations_{t}"] = len(s)
        rows.append(row)
    return pd.DataFrame(rows).set_index("run")


def plot_station_bias(stats, runs, min_count, title, path):
    """Biais moyen par station (une ligne par station, triées par distance), P et S côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))
    for ax, t in zip(axes, ("P", "S")):
        s = stats[stats["phase_type"] == t]
        enough = s[s["n"] >= min_count].groupby("station")["author"].nunique()
        stations = enough[enough == len(runs)].index
        order = (s[s["station"].isin(stations)].groupby("station")["distance_mediane_km"]
                 .median().sort_values().index)
        y = {st: i for i, st in enumerate(order)}
        ax.axvline(0, color=INK_MUTED, linewidth=1)
        for i, author in enumerate(runs):
            r = s[(s["author"] == author) & s["station"].isin(order)]
            ax.scatter(r["moyenne_s"], r["station"].map(y) + (i - (len(runs) - 1) / 2) * 0.18,
                       s=18, color=SERIES[i], label=author, zorder=3)
        ax.set_yticks(range(len(order)), [f"{st} ({s[s['station'] == st]['distance_mediane_km'].median():.0f} km)"
                                          for st in order])
        ax.invert_yaxis()
        ax.set_xlabel(f"Résidu moyen {t} (s) — positif : arrivée plus tardive que prévu")
        ax.set_title(f"Biais par station, ondes {t}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=len(runs), bbox_to_anchor=(0.01, -0.03))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.99, 0.005, f"Stations avec au moins {min_count} résidus dans chaque run, "
             "triées par distance médiane", ha="right", fontsize=7, color=INK_MUTED)
    save(fig, path)


def plot_residual_vs_distance(phases, runs, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    centers = (np.array(DISTANCE_BINS_KM[:-1]) + np.array(DISTANCE_BINS_KM[1:])) / 2
    for ax, t in zip(axes, ("P", "S")):
        ax.axhline(0, color=INK_MUTED, linewidth=1)
        for i, author in enumerate(runs):
            p = phases[(phases["author"] == author) & (phases["phase_type"] == t)]
            b = pd.cut(p["dist_km"], DISTANCE_BINS_KM)
            med = p.groupby(b, observed=False)["residual_s"].median().to_numpy()
            ax.plot(centers, med, marker="o", markersize=3, color=SERIES[i], label=author)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
        ax.set_xlabel("Distance épicentrale (km)")
        ax.set_ylabel(f"Résidu médian {t} (s)")
        ax.set_title(f"Ondes {t}")
    axes[0].legend(loc="best")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_station_residuals")

    runs = list(require(config, "compare.nll_authors"))
    classes = require(config, "nll.quality_classes")
    good = config["compare"].get("well_constrained_classes") or ["A", "B"]
    min_count = config.get("residuals", {}).get("min_count", 50)

    phases = load_nll_phases(config, runs)
    runs = [r for r in runs if r in set(phases["author"])]
    if not runs:
        raise ValueError("Aucun run de compare.nll_authors n'a de phases (runs locaux seulement)")
    log(f"{len(runs)} runs : {runs}")

    bulletin = load_bulletin(config)
    origins = {r: nll_origins(bulletin, r) for r in runs}
    subsets = quality_subsets(origins, classes, good)
    phases = phases[phases["weight"] > 0]

    all_summaries = []
    for name, events in subsets.items():
        sub_out = out / name
        sub_out.mkdir(exist_ok=True)
        ph = phases[phases["event_id"].isin(events)]
        log(f"Sous-ensemble {name} : {len(events)} événements, {len(ph)} phases")
        stats = station_stats(ph)
        summary = run_summary(ph, stats, min_count)
        log(summary.round(3).to_string(), 1)
        stats.to_csv(sub_out / "stations.csv", index=False)
        summary.to_csv(sub_out / "summary.csv")
        all_summaries.append(summary.assign(sous_ensemble=name))
        label = f"{name} — {len(events)} événements communs"
        plot_station_bias(stats, runs, min_count, f"Résidus par station ({label})",
                          sub_out / "station_bias.png")
        plot_residual_vs_distance(ph, runs, f"Résidu médian selon la distance ({label})",
                                  sub_out / "residual_vs_distance.png")
    pd.concat(all_summaries).to_csv(out / "summary_all.csv")


if __name__ == "__main__":
    main()
