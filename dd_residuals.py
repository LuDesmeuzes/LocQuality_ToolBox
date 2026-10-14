"""
dd_residuals.py
===============
Résidus avant / après relocalisation scrtdd :
  - RMS des résidus double différence par événement (startRms → finalRms,
    reloc-event.csv ; ce ne sont PAS des résidus absolus)
  - résidus absolus par phase, globalement et par station (reloc-phase.csv)
  - résidus double différence finaux en fonction de la distance
    inter-événements, catalogue vs corrélation croisée
    (cluster-N-final-double-difference.csv)

Sorties (dans <output_dir>/dd_residuals/) :
  - station_residuals.csv : moyenne / écart-type des résidus par station et
                            phase, avant et après (repérer les stations
                            biaisées → corrections de station)
  - rms.png               : distributions des RMS avant / après
  - station_residuals.png : résidu moyen final par station et par phase
  - dd_residuals.png      : résidus DD finaux vs distance inter-événements

Utilisation :
    python dd_residuals.py
    python dd_residuals.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SERIES, INK_MUTED, ScrtddRun, hist_compare, load_config, log, output_dir,
    parse_args, require, save, setup_style,
)


def plot_rms(ev, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    top = float(np.nanpercentile(np.r_[ev["startRms"], ev["finalRms"]], 99))
    bins = np.linspace(0, top, 61)
    hist_compare(axes[0], {"Avant (startRms)": ev["startRms"].clip(upper=top),
                           "Après (finalRms)": ev["finalRms"].clip(upper=top)},
                 bins, "RMS des résidus DD par événement (s)")
    axes[0].set_title(f"RMS DD avant / après (n={len(ev)})")

    ax = axes[1]
    ax.scatter(ev["startRms"], ev["finalRms"], s=3, color=SERIES[0], alpha=0.4, linewidths=0)
    ax.plot([0, top], [0, top], color=INK_MUTED, linestyle="--", linewidth=1,
            label=f"{(ev['finalRms'] < ev['startRms']).mean() * 100:.0f} % des événements "
                  "sous la diagonale (RMS amélioré)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("RMS DD avant — startRms (s)")
    ax.set_ylabel("RMS DD après — finalRms (s)")
    ax.set_title("RMS après vs avant")
    ax.legend(loc="upper left")
    save(fig, path)


def station_table(ph):
    used = ph[ph["usedInReloc"].astype(str).str.lower() == "true"].copy()
    used["phase"] = used["type"].str[0].str.upper()
    used["station"] = used["networkCode"] + "." + used["stationCode"]
    tab = (used.groupby(["station", "phase"])
               .agg(n=("finalResidual", "size"),
                    start_mean=("startResidual", "mean"),
                    start_std=("startResidual", "std"),
                    final_mean=("finalResidual", "mean"),
                    final_std=("finalResidual", "std"))
               .reset_index())
    return used, tab


def plot_station_residuals(tab, path, min_obs):
    tab = tab[tab["n"] >= min_obs]
    stations = (tab.groupby("station")["n"].sum().sort_values(ascending=False).index)
    y = {s: i for i, s in enumerate(stations)}
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(stations) + 1.5)))
    for k, (phase, offset) in enumerate([("P", -0.15), ("S", 0.15)]):
        sub = tab[tab["phase"] == phase]
        ys = sub["station"].map(y) + offset
        ax.errorbar(sub["final_mean"], ys, xerr=sub["final_std"], fmt="o", markersize=4,
                    color=SERIES[k], ecolor=SERIES[k], elinewidth=1, capsize=0,
                    label=f"{phase} (moyenne ± écart-type)")
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations)
    ax.invert_yaxis()
    ax.set_xlabel("Résidu final (s)")
    ax.set_title(f"Résidus finaux par station (≥ {min_obs} observations)")
    ax.legend(loc="upper right")
    save(fig, path)


def plot_dd_residuals(dd, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    dd = dd[dd["weight"] > 0]
    groups = {"Catalogue": dd[~dd["xcorrUsed"].astype(bool)],
              "Corrélation croisée": dd[dd["xcorrUsed"].astype(bool)]}
    groups = {k: v for k, v in groups.items() if len(v)}

    bins = np.linspace(-0.5, 0.5, 101)
    hist_compare(axes[0], {k: v["doubleDifferenceResidual"].clip(-0.5, 0.5)
                           for k, v in groups.items()},
                 bins, "Résidu double différence final (s)")
    axes[0].set_title("Résidus DD finaux (poids > 0)")

    ax = axes[1]
    edges = np.quantile(dd["interEventDistance"], np.linspace(0, 1, 21))
    for i, (label, sub) in enumerate(groups.items()):
        idx = np.clip(np.digitize(sub["interEventDistance"], edges) - 1, 0, len(edges) - 2)
        g = sub["doubleDifferenceResidual"].abs().groupby(idx)
        centers = sub["interEventDistance"].groupby(idx).median()
        med = g.median()
        p84 = g.quantile(0.84)
        ax.plot(centers, med, color=SERIES[i], marker="o", markersize=4, label=f"{label} — médiane")
        ax.plot(centers, p84, color=SERIES[i], linestyle="--", linewidth=1,
                label=f"{label} — 84e percentile")
    ax.set_xlabel("Distance inter-événements (km)")
    ax.set_ylabel("|Résidu DD final| (s)")
    ax.set_title("Résidus DD vs distance inter-événements")
    ax.legend(loc="upper left")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "dd_residuals")

    run_dir = require(config, "paths.scrtdd_run_dir")
    log(f"Lecture du run scrtdd {run_dir}")
    run = ScrtddRun(run_dir, config["paths"].get("scrtdd_eventlist"))

    ev = run.reloc_events()
    log(f"RMS DD médian : avant {ev['startRms'].median():.4f} s, "
        f"après {ev['finalRms'].median():.4f} s", 1)

    used, tab = station_table(run.reloc_phases())
    for col in ["startResidual", "finalResidual"]:
        for phase, g in used.groupby("phase"):
            log(f"{col} {phase} : médiane |rés.| {g[col].abs().median():.4f} s "
                f"(n={len(g)})", 1)
    tab.to_csv(out / "station_residuals.csv", index=False)
    log(f"→ {out / 'station_residuals.csv'}", 1)

    log("Lecture des résidus double différence (peut prendre un moment)")
    dd = run.double_differences("final")
    log(f"{len(dd)} doubles différences finales", 1)

    log("Figures")
    plot_rms(ev, out / "rms.png")
    plot_station_residuals(tab, out / "station_residuals.png", min_obs=20)
    if not dd.empty:
        plot_dd_residuals(dd, out / "dd_residuals.png")


if __name__ == "__main__":
    main()
