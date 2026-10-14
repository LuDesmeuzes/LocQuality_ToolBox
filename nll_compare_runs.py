"""
nll_compare_runs.py
===================
Compare plusieurs runs NLL présents dans le même bulletin SeisComP purifié
(colonne Author : 1_ubaye_v3, 5_ubaye_v3, 025_ubaye_v2, …), par exemple
pour choisir un pas de grille ou un modèle de vitesse.

Les indicateurs sont calculés pour chaque run sur tous ses événements, et
aussi sur les événements communs à tous les runs (pour que les écarts ne
viennent pas d'échantillons différents). Pour chaque run :
RMS, incertitudes, répartition en classes de qualité (nll.quality_classes)
et écart à l'origine de référence (LOCSAT).

Sorties (dans <output_dir>/nll_compare_runs/) :
  - runs.csv            : un indicateur par colonne, une ligne par run, calculés
                          sur tous ses événements et sur les événements communs
  - compare.png         : médiane et intervalle 16–84 % du RMS, de l'incertitude
                          horizontale et de l'écart à la référence, par run
  - compare_common.png  : idem sur les seuls événements communs

Utilisation :
    python nll_compare_runs.py
    python nll_compare_runs.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import pandas as pd

from loc_common import (
    NLL_METHOD, SERIES, classify_quality, load_config, log,
    nll_origins, output_dir, pair_with_reference, parse_args, read_bulletin,
    reference_origins, require, save, setup_style,
)


def run_row(df, ref, classes):
    df = df.copy()
    df["quality"] = classify_quality(df, classes)
    p = pair_with_reference(df, ref)
    row = {"n": len(df),
           "rms_median_s": df["rms"].median(),
           "erh_median_km": df["erh_km"].median(),
           "erz_median_km": df["depth_unc_km"].median(),
           "dh_ref_median_km": p["dh_km"].median(),
           "dz_ref_median_km": p["dz_km"].median(),
           "d3_ref_median_km": p["d3_km"].median()}
    for c in classes:
        row[f"pct_{c['name']}"] = 100 * (df["quality"] == c["name"]).mean()
    return row, p


def plot_dots(pairs, path, subtitle):
    """Un run par ligne : médiane (point) et intervalle 16–84 % (trait)."""
    order = sorted(pairs, key=lambda a: pairs[a]["d3_km"].median(), reverse=True)
    specs = [("rms", "RMS (s)"), ("erh_km", "Incertitude horizontale erh (km)"),
             ("d3_km", "Écart 3D à la référence (km)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, 0.32 * len(order) + 1.5)), sharey=True)
    for ax, (col, label) in zip(axes, specs):
        for i, a in enumerate(order):
            v = pairs[a][col].dropna()
            ax.plot([v.quantile(0.16), v.quantile(0.84)], [i, i], color=SERIES[0],
                    linewidth=2, alpha=0.35, solid_capstyle="round")
            ax.plot(v.median(), i, "o", color=SERIES[0], markersize=6)
        ax.set_xlabel(label)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order)
    axes[0].set_title(f"Runs NLL — médiane et intervalle 16–84 % ({subtitle})")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_compare_runs")
    classes = require(config, "nll.quality_classes")

    log(f"Lecture du bulletin {require(config, 'paths.bulletin')}")
    bulletin = read_bulletin(config["paths"]["bulletin"])
    authors = (config.get("compare", {}).get("nll_authors")
               or sorted(bulletin.loc[bulletin["method"] == NLL_METHOD, "author"].dropna().unique()))
    log(f"{len(authors)} runs comparés : {authors}", 1)

    ref = reference_origins(bulletin)
    runs = {a: nll_origins(bulletin, a) for a in authors}
    common = set.intersection(*(set(df["event_id"]) for df in runs.values()))
    log(f"{len(common)} événements communs à tous les runs", 1)

    rows, pairs = {}, {}
    for a, df in runs.items():
        rows[(a, "tous")], pairs[a] = run_row(df, ref, classes)
        if common:
            rows[(a, "communs")], _ = run_row(df[df["event_id"].isin(common)], ref, classes)

    table = pd.DataFrame(rows).T
    table.index.names = ["run", "événements"]
    log(table.round(3).to_string(), 1)
    table.to_csv(out / "runs.csv")
    log(f"→ {out / 'runs.csv'}", 1)

    log("Figures")
    plot_dots(pairs, out / "compare.png", "chaque run sur tous ses événements")
    if common and len(common) >= 50:
        plot_dots({a: p[p["event_id"].isin(common)] for a, p in pairs.items()},
                  out / "compare_common.png", f"{len(common)} événements communs")
    else:
        log("Moins de 50 événements communs : pas de figure sur les événements communs "
            "(restreindre compare.nll_authors)", 1)


if __name__ == "__main__":
    main()
