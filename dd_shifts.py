"""
dd_shifts.py
============
Déplacements induits par la relocalisation double différence (scrtdd) :
position initiale (input-event.csv, c.-à-d. l'origine NLL passée à scrtdd)
→ position relocalisée (reloc-event.csv).

Sorties (dans <output_dir>/dd_shifts/) :
  - shifts.csv        : une ligne par événement relocalisé, avec le déplacement
  - not_relocated.csv : événements d'entrée absents de reloc-event.csv
  - shifts.png        : distributions des déplacements horizontaux et
                        verticaux, nuage est-nord
  - map_shifts.png    : carte et coupes, initial → relocalisé (vecteurs)
  - map_before.png / map_after.png : carte et coupes avant / après, séparées

Utilisation :
    python dd_shifts.py
    python dd_shifts.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SERIES, INK_MUTED, ScrtddRun, load_config, log, map_and_sections,
    output_dir, parse_args, require, save, setup_style,
)


def plot_shifts(df, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.hist(df["dh_km"].clip(upper=5), bins=np.linspace(0, 5, 51),
            color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(df["dh_km"].median(), color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Déplacement horizontal (km)")
    ax.set_ylabel("Nombre d'événements")
    ax.set_title(f"{title} (n={len(df)})")

    ax = axes[1]
    ax.hist(df["dz_km"].clip(-5, 5), bins=np.linspace(-5, 5, 81),
            color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(df["dz_km"].median(), color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Déplacement vertical (km, + = plus profond après reloc.)")
    ax.set_ylabel("Nombre d'événements")

    ax = axes[2]
    ax.scatter(df["de_km"], df["dn_km"], s=3, color=SERIES[0], alpha=0.4, linewidths=0)
    lim = max(1.0, float(np.nanpercentile(df["dh_km"], 95)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.plot(df["de_km"].median(), df["dn_km"].median(), marker="+", markersize=14,
            color="#0b0b0b", markeredgewidth=2, label="Médiane (translation d'ensemble)")
    ax.set_xlabel("Déplacement est (km)")
    ax.set_ylabel("Déplacement nord (km)")
    ax.legend(loc="upper right")
    fig.text(0.99, 0.01, "Valeurs hors échelle regroupées dans les barres extrêmes",
             ha="right", fontsize=7, color=INK_MUTED)
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "dd_shifts")

    run_dir = require(config, "paths.scrtdd_run_dir")
    log(f"Lecture du run scrtdd {run_dir}")
    run = ScrtddRun(run_dir, config["paths"].get("scrtdd_eventlist"))
    inp = run.input_events()
    df = run.shifts()
    lost = inp[~inp["id"].isin(df["id"])]
    log(f"{len(inp)} événements en entrée, {len(df)} relocalisés "
        f"({100 * len(df) / max(len(inp), 1):.0f} %), {len(lost)} non relocalisés", 1)

    stats = df[["dh_km", "dz_km", "de_km", "dn_km"]].describe(
        percentiles=[0.16, 0.5, 0.84, 0.95]).T
    log(stats.round(3).to_string(), 1)

    df.to_csv(out / "shifts.csv", index=False)
    lost.to_csv(out / "not_relocated.csv", index=False)
    stats.to_csv(out / "shifts_stats.csv")
    log(f"→ {out / 'shifts.csv'}", 1)

    log("Figures")
    plot_shifts(df, out / "shifts.png", "Déplacements scrtdd")

    map_cfg = config.get("map", {})
    stations = run.stations() if map_cfg.get("show_stations") else None
    initial = {"lon": df["longitude_init"], "lat": df["latitude_init"],
               "depth": df["depth_init"], "label": "Initial (NLL)", "color": SERIES[1]}
    reloc = {"lon": df["longitude"], "lat": df["latitude"], "depth": df["depth"],
             "label": "Relocalisé", "color": SERIES[0]}
    map_and_sections(
        [initial, reloc], map_cfg.get("sections"), "Initial → relocalisé",
        out / "map_shifts.png",
        vectors={"lon0": df["longitude_init"], "lat0": df["latitude_init"],
                 "depth0": df["depth_init"], "lon1": df["longitude"],
                 "lat1": df["latitude"], "depth1": df["depth"]},
        stations=stations, zone=map_cfg.get("zone"))
    # Les deux nuages séparément : plus lisible que la superposition
    map_and_sections([initial], map_cfg.get("sections"), "Avant relocalisation",
                     out / "map_before.png", stations=stations, zone=map_cfg.get("zone"))
    map_and_sections([reloc], map_cfg.get("sections"), "Après relocalisation",
                     out / "map_after.png", stations=stations, zone=map_cfg.get("zone"))


if __name__ == "__main__":
    main()
