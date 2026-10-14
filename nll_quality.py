"""
nll_quality.py
==============
Indicateurs de qualité des localisations NLL d'un run (nll.author) lues dans
le bulletin SeisComP purifié, et classement de chaque origine en classes de
qualité (nll.quality_classes).

Indicateurs utilisés : RMS (Quality Standard Error), incertitude horizontale
(erh = max des incertitudes en latitude et longitude) et incertitude en
profondeur. Si le bulletin contient aussi des colonnes gap / nphs / dmin_km,
les critères correspondants sont appliqués.

Sorties (dans <output_dir>/nll_quality/) :
  - nll_quality.csv          : une ligne par origine NLL, avec sa classe
  - nll_quality_stats.csv    : statistiques par classe
  - distributions.png        : histogrammes RMS / erh / erz
  - map_classes.png          : carte et coupes colorées par classe

Utilisation :
    python nll_quality.py
    python nll_quality.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SEQ_BLUE, INK_MUTED, classify_quality, load_config, log, map_and_sections,
    nll_origins, output_dir, parse_args, read_bulletin, require, save, setup_style,
)


def plot_distributions(df, path, author):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = [("rms", "RMS (s)", np.linspace(0, 0.3, 61)),
             ("erh_km", "Incertitude horizontale erh (km)", np.linspace(0, 10, 51)),
             ("depth_unc_km", "Incertitude en profondeur (km)", np.linspace(0, 10, 51))]
    for ax, (col, label, bins) in zip(axes, specs):
        values = df[col].dropna()
        ax.hist(values.clip(upper=bins[-1]), bins=bins, color=SEQ_BLUE[2],
                edgecolor="#fcfcfb", linewidth=0.5)
        med = values.median()
        ax.axvline(med, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.annotate(f"médiane {med:.3g}", (med, ax.get_ylim()[1] * 0.95),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_MUTED)
        ax.set_xlabel(label)
        ax.set_ylabel("Nombre d'origines")
    axes[0].set_title(f"Qualité NLL — {author} (n={len(df)})")
    fig.text(0.99, 0.01, "Valeurs hors échelle regroupées dans la dernière barre",
             ha="right", fontsize=7, color=INK_MUTED)
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_quality")

    author = require(config, "nll.author")
    classes = require(config, "nll.quality_classes")

    log(f"Lecture du bulletin {require(config, 'paths.bulletin')}")
    bulletin = read_bulletin(config["paths"]["bulletin"])
    nll = nll_origins(bulletin, author)
    log(f"{len(nll)} origines NLL pour le run {author}", 1)

    log("Classement en classes de qualité")
    nll["quality"] = classify_quality(nll, classes)
    order = [c["name"] for c in classes]

    stats = (nll.groupby("quality")
                .agg(n=("event_id", "size"),
                     rms_median=("rms", "median"),
                     erh_median_km=("erh_km", "median"),
                     erz_median_km=("depth_unc_km", "median"),
                     depth_median_km=("depth", "median"))
                .reindex(order).dropna(how="all"))
    stats["pourcentage"] = 100 * stats["n"] / stats["n"].sum()
    log(stats.round(3).to_string(), 1)

    nll.to_csv(out / "nll_quality.csv", index=False)
    stats.to_csv(out / "nll_quality_stats.csv")
    log(f"→ {out / 'nll_quality.csv'}", 1)

    log("Figures")
    plot_distributions(nll, out / "distributions.png", author)

    # Classes ordinales → rampe séquentielle (foncé = meilleure qualité),
    # la moins bonne classe tracée en premier pour rester sous les autres
    ramp = SEQ_BLUE[::-1] + [SEQ_BLUE[0]] * max(0, len(order) - len(SEQ_BLUE))
    layers = []
    for name, color in reversed(list(zip(order, ramp))):
        sub = nll[nll["quality"] == name]
        layers.append({"lon": sub["longitude"], "lat": sub["latitude"],
                       "depth": sub["depth"], "label": f"Classe {name}",
                       "color": color, "size": 4})
    map_cfg = config.get("map", {})
    map_and_sections(layers, map_cfg.get("sections"),
                     f"Classes de qualité NLL — {author}",
                     out / "map_classes.png", zone=map_cfg.get("zone"))


if __name__ == "__main__":
    main()
