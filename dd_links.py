"""
dd_links.py
===========
Bilan des liens double différence d'un run scrtdd et contrôle de la
corrélation croisée :
  - par événement : nombre de doubles différences utilisées (poids > 0),
    séparées catalogue / corrélation croisée et P / S, nombre de voisins
  - taille des clusters et événements non relocalisés
  - corrélation croisée (xcorr.csv) : distribution des coefficients et des
    décalages (lags). Une forte proportion de lags en bord de fenêtre de
    recherche signale des mesures non fiables.

Sorties (dans <output_dir>/dd_links/) :
  - event_links.csv  : une ligne par événement relocalisé
  - clusters.csv     : nombre d'événements par cluster
  - links.png        : distributions du nombre de DD et de voisins par événement
  - xcorr.png        : coefficients et lags de corrélation croisée

Utilisation :
    python dd_links.py
    python dd_links.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SERIES, INK_MUTED, ScrtddRun, hist_compare, load_config, log, output_dir,
    parse_args, require, save, setup_style,
)


def event_links(dd):
    """Compte par événement les DD utilisées, par type de donnée et de phase."""
    dd = dd[dd["weight"] > 0].copy()
    dd["source"] = np.where(dd["xcorrUsed"].astype(bool), "cc", "ct")
    dd["ptype"] = dd["phaseType"].str[0].str.upper()
    # Chaque DD concerne deux événements : on la compte pour chacun
    both = pd.concat([
        dd[["eventId1", "eventId2", "source", "ptype"]].rename(
            columns={"eventId1": "id", "eventId2": "other"}),
        dd[["eventId2", "eventId1", "source", "ptype"]].rename(
            columns={"eventId2": "id", "eventId1": "other"}),
    ])
    counts = both.groupby(["id", "source", "ptype"]).size().unstack(["source", "ptype"],
                                                                     fill_value=0)
    counts.columns = [f"n_{s}_{p}" for s, p in counts.columns]
    counts["n_dd"] = counts.sum(axis=1)
    counts["n_neighbours"] = both.groupby("id")["other"].nunique()
    return counts.reset_index()


def plot_links(links, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ct = links.filter(like="n_ct_").sum(axis=1)
    cc = links.filter(like="n_cc_").sum(axis=1)
    top = float(np.nanpercentile(np.r_[ct, cc], 98))
    hist_compare(axes[0], {"Catalogue": ct.clip(upper=top), "Corrélation croisée": cc.clip(upper=top)},
                 np.linspace(0, top, 51), "Doubles différences utilisées par événement")
    axes[0].set_title(f"Observations DD par événement (n={len(links)})")
    top = float(np.nanpercentile(links["n_neighbours"], 98))
    axes[1].hist(links["n_neighbours"].clip(upper=top), bins=np.linspace(0, top, 51),
                 color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5)
    axes[1].axvline(links["n_neighbours"].median(), color=INK_MUTED, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Nombre d'événements voisins liés")
    axes[1].set_ylabel("Nombre d'événements")
    axes[1].set_title("Voisins par événement")
    save(fig, path)


def plot_xcorr(xc, max_lag, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ptype = xc["phaseType"].str[0].str.upper()
    hist_compare(axes[0], {f"{p}": xc.loc[ptype == p, "coefficient"].abs() for p in ["P", "S"]},
                 np.linspace(0, 1, 51), "|Coefficient de corrélation|")
    axes[0].set_title(f"Corrélation croisée (mesures valides, n={len(xc)})")

    ax = axes[1]
    bins = np.linspace(-max_lag, max_lag, 101)
    for i, p in enumerate(["P", "S"]):
        ax.hist(xc.loc[ptype == p, "lag"], bins=bins, histtype="step", linewidth=1.5,
                color=SERIES[i], label=p)
    edge = (xc["lag"].abs() >= 0.98 * max_lag).mean() * 100
    ax.set_xlabel("Lag (s)")
    ax.set_ylabel("Nombre de mesures")
    ax.set_title(f"Lags — {edge:.1f} % en bord de fenêtre (|lag| ≥ 98 % du max)")
    ax.legend(loc="upper right")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "dd_links")

    run_dir = require(config, "paths.scrtdd_run_dir")
    log(f"Lecture du run scrtdd {run_dir}")
    run = ScrtddRun(run_dir, config["paths"].get("scrtdd_eventlist"))
    inp = run.input_events()
    reloc = run.reloc_events()

    clusters = run.event_clusters()
    sizes = clusters.value_counts().sort_index().rename("n_events")
    sizes.index.name = "cluster"
    log(f"{len(sizes)} clusters, tailles : {sizes.to_dict()}", 1)
    log(f"{len(inp) - len(clusters)} événements d'entrée hors de tout cluster "
        f"(isolés), {len(clusters) - len(reloc)} en cluster mais non relocalisés", 1)
    sizes.to_csv(out / "clusters.csv")

    log("Comptage des liens double différence")
    links = event_links(run.double_differences("final"))
    links["cluster"] = links["id"].map(clusters)
    links = reloc[["id", "time", "latitude", "longitude", "depth"]].merge(links, on="id", how="left")
    log(f"Médiane par événement : {links['n_dd'].median():.0f} DD, "
        f"{links['n_neighbours'].median():.0f} voisins", 1)
    links.to_csv(out / "event_links.csv", index=False)
    log(f"→ {out / 'event_links.csv'}", 1)

    log("Figures")
    plot_links(links.dropna(subset=["n_dd"]), out / "links.png")

    xc = run.xcorr()
    if xc.empty:
        log("Pas de xcorr.csv (run sans corrélation croisée)", 1)
        return
    valid = xc[xc["valid"].astype(bool)]
    max_lag = float(valid["lag"].abs().max())
    edge = (valid["lag"].abs() >= 0.98 * max_lag)
    log(f"Corrélation croisée : {len(xc)} mesures, {len(valid)} valides ; "
        f"lag max {max_lag:.2f} s ; {edge.mean() * 100:.1f} % en bord de fenêtre ; "
        f"|coef| médian {valid['coefficient'].abs().median():.2f}", 1)
    plot_xcorr(valid, max_lag, out / "xcorr.png")


if __name__ == "__main__":
    main()
