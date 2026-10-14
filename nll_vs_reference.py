"""
nll_vs_reference.py
===================
Compare les localisations NLL d'un run (nll.author) à l'origine de
référence de chaque événement (origine préférée non NLL, en pratique la
localisation manuelle LOCSAT), à partir du bulletin SeisComP purifié.

Pour chaque événement présent dans les deux : écart horizontal (est, nord),
écart en profondeur (positif = NLL plus profond), écart 3D et écart en temps
origine. Les écarts sont aussi rapportés aux classes de qualité NLL.

Sorties (dans <output_dir>/nll_vs_reference/) :
  - pairs.csv          : une ligne par événement (NLL + référence + écarts)
  - stats.csv          : médianes et percentiles des écarts, par classe NLL
  - shifts.png         : distributions des écarts horizontaux / verticaux,
                         nuage est-nord des écarts
  - map_shifts.png     : carte et coupes, référence → NLL
  - shift_vs_unc.png   : écart horizontal en fonction de l'incertitude NLL

Utilisation :
    python nll_vs_reference.py
    python nll_vs_reference.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SERIES, INK_MUTED, classify_quality, load_config, log, map_and_sections,
    nll_origins, output_dir, pair_with_reference, parse_args, read_bulletin,
    reference_origins, require, save, setup_style,
)


def describe(values):
    v = pd.Series(values).dropna()
    return pd.Series({"n": len(v), "median": v.median(), "p16": v.quantile(0.16),
                      "p84": v.quantile(0.84), "p95": v.quantile(0.95)})


def plot_shifts(pairs, path, author):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.hist(pairs["dh_km"].clip(upper=10), bins=np.linspace(0, 10, 51),
            color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(pairs["dh_km"].median(), color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Écart horizontal NLL − référence (km)")
    ax.set_ylabel("Nombre d'événements")
    ax.set_title(f"NLL {author} vs référence (n={len(pairs)})")

    ax = axes[1]
    ax.hist(pairs["dz_km"].clip(-10, 10), bins=np.linspace(-10, 10, 81),
            color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(pairs["dz_km"].median(), color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Écart en profondeur NLL − référence (km, + = NLL plus profond)")
    ax.set_ylabel("Nombre d'événements")

    ax = axes[2]
    ax.scatter(pairs["de_km"], pairs["dn_km"], s=3, color=SERIES[0], alpha=0.4, linewidths=0)
    lim = max(2.0, float(np.nanpercentile(pairs["dh_km"], 95)))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=INK_MUTED, linewidth=0.8)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8)
    ax.plot(np.median(pairs["de_km"]), np.median(pairs["dn_km"]), marker="+",
            markersize=14, color="#0b0b0b", markeredgewidth=2, label="Médiane (biais)")
    ax.set_xlabel("Écart est (km)")
    ax.set_ylabel("Écart nord (km)")
    ax.legend(loc="upper right")
    fig.text(0.99, 0.01, "Valeurs hors échelle regroupées dans les barres extrêmes",
             ha="right", fontsize=7, color=INK_MUTED)
    save(fig, path)


def plot_shift_vs_unc(pairs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, unc, shift, label in [
        (axes[0], "erh_km", "dh_km", "horizontal"),
        (axes[1], "depth_unc_km", "dz_km", "en profondeur"),
    ]:
        y = pairs[shift].abs()
        ax.scatter(pairs[unc], y, s=3, color=SERIES[0], alpha=0.3, linewidths=0)
        top = float(np.nanpercentile(np.r_[pairs[unc], y], 98))
        frac = (y <= pairs[unc]).mean() * 100
        ax.plot([0, top], [0, top], color=INK_MUTED, linewidth=1, linestyle="--",
                label=f"écart = incertitude ({frac:.0f} % des événements en dessous)")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_title(f"Écart {label} vs incertitude")
        ax.set_xlabel(f"Incertitude NLL {label} (km)")
        ax.set_ylabel(f"|Écart {label}| NLL − référence (km)")
        ax.legend(loc="upper left")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_vs_reference")
    author = require(config, "nll.author")

    log(f"Lecture du bulletin {require(config, 'paths.bulletin')}")
    bulletin = read_bulletin(config["paths"]["bulletin"])
    nll = nll_origins(bulletin, author)
    ref = reference_origins(bulletin)
    pairs = pair_with_reference(nll, ref)
    log(f"{len(nll)} origines NLL, {len(ref)} origines de référence, "
        f"{len(pairs)} événements communs", 1)
    log("Origines de référence par méthode / auteur : "
        + ", ".join(f"{k} {v}" for k, v in pairs["method_ref"].value_counts().items()), 1)

    pairs["quality"] = classify_quality(pairs, require(config, "nll.quality_classes"))

    rows = {}
    for name, sub in [("tous", pairs)] + list(pairs.groupby("quality")):
        for col in ["dh_km", "dz_km", "d3_km", "dt_s"]:
            rows[(name, col)] = describe(sub[col])
    stats = pd.DataFrame(rows).T
    stats.index.names = ["classe", "écart"]
    log(stats.round(2).to_string(), 1)

    pairs.to_csv(out / "pairs.csv", index=False)
    stats.to_csv(out / "stats.csv")
    log(f"→ {out / 'pairs.csv'}", 1)

    log("Figures")
    plot_shifts(pairs, out / "shifts.png", author)
    plot_shift_vs_unc(pairs, out / "shift_vs_unc.png")
    map_cfg = config.get("map", {})
    map_and_sections(
        [{"lon": pairs["longitude_ref"], "lat": pairs["latitude_ref"],
          "depth": pairs["depth_ref"], "label": "Référence", "color": SERIES[1]},
         {"lon": pairs["longitude"], "lat": pairs["latitude"],
          "depth": pairs["depth"], "label": f"NLL {author}", "color": SERIES[0]}],
        map_cfg.get("sections"), "Référence → NLL", out / "map_shifts.png",
        vectors={"lon0": pairs["longitude_ref"], "lat0": pairs["latitude_ref"],
                 "depth0": pairs["depth_ref"], "lon1": pairs["longitude"],
                 "lat1": pairs["latitude"], "depth1": pairs["depth"]},
        zone=map_cfg.get("zone"))


if __name__ == "__main__":
    main()
