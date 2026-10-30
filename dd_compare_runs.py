"""
dd_compare_runs.py
==================
Compare plusieurs runs scrtdd (compare.scrtdd_runs : étiquette → dossier),
typiquement le même catalogue relocalisé avec des paramètres différents.

Pour chaque run : taux de relocalisation, déplacements médians, RMS DD
avant / après, résidus absolus finaux et, s'il y a de la corrélation
croisée, proportion de lags en bord de fenêtre.

Sorties (dans <output_dir>/dd_compare_runs/) :
  - runs.csv          : un indicateur par colonne, un run par ligne
  - compare.png       : distributions des déplacements et des RMS DD par run
  - map_<run>.png     : carte et coupes de chaque run relocalisé
  - map_<run>_temps.png : idem, événements colorés par date
  - profondeur_temps.png : profondeur des événements relocalisés en fonction
                        du temps, un panneau par run (migration verticale)

Utilisation :
    python dd_compare_runs.py
    python dd_compare_runs.py --config un_autre_config.yaml
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    SERIES, ScrtddRun, hist_compare, load_config, log, map_and_sections,
    output_dir, parse_args, require, save, setup_style,
)


def run_indicators(run):
    inp = run.input_events()
    sh = run.shifts()
    ph = run.reloc_phases()
    used = ph[ph["usedInReloc"].astype(str).str.lower() == "true"]
    ptype = used["type"].str[0].str.upper()
    row = {
        "n_input": len(inp),
        "n_relocated": len(sh),
        "pct_relocated": 100 * len(sh) / max(len(inp), 1),
        "n_clusters": len(run.cluster_files("event")),
        "dh_median_km": sh["dh_km"].median(),
        "dh_p84_km": sh["dh_km"].quantile(0.84),
        "dz_median_km": sh["dz_km"].median(),
        "de_median_km": sh["de_km"].median(),
        "dn_median_km": sh["dn_km"].median(),
        "rms_dd_start_median_s": sh["startRms"].median(),
        "rms_dd_final_median_s": sh["finalRms"].median(),
        "abs_res_P_final_median_s": used.loc[ptype == "P", "finalResidual"].abs().median(),
        "abs_res_S_final_median_s": used.loc[ptype == "S", "finalResidual"].abs().median(),
    }
    xc = run.xcorr()
    if not xc.empty:
        valid = xc[xc["valid"].astype(bool)]
        max_lag = valid["lag"].abs().max()
        row["xcorr_valid"] = len(valid)
        row["xcorr_pct_lag_edge"] = 100 * (valid["lag"].abs() >= 0.98 * max_lag).mean()
        row["xcorr_coef_median"] = valid["coefficient"].abs().median()
    return row, sh


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "dd_compare_runs")

    runs = require(config, "compare.scrtdd_runs")
    if len(runs) > len(SERIES):
        raise ValueError(f"Au plus {len(SERIES)} runs comparables sur une même figure")

    rows, shifts = {}, {}
    for label, run_dir in runs.items():
        log(f"Run {label} : {run_dir}")
        run = ScrtddRun(run_dir)
        rows[label], shifts[label] = run_indicators(run)

    table = pd.DataFrame(rows).T
    table.index.name = "run"
    log(table.T.round(3).to_string(), 1)
    table.to_csv(out / "runs.csv")
    log(f"→ {out / 'runs.csv'}", 1)

    log("Figures")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    hist_compare(axes[0], {k: v["dh_km"].clip(upper=5) for k, v in shifts.items()},
                 np.linspace(0, 5, 51), "Déplacement horizontal (km)")
    axes[0].set_title("Déplacements horizontaux")
    hist_compare(axes[1], {k: v["dz_km"].clip(-5, 5) for k, v in shifts.items()},
                 np.linspace(-5, 5, 81), "Déplacement vertical (km, + = plus profond)")
    axes[1].set_title("Déplacements verticaux")
    top = float(np.nanpercentile(pd.concat([v["finalRms"] for v in shifts.values()]), 99))
    hist_compare(axes[2], {k: v["finalRms"].clip(upper=top) for k, v in shifts.items()},
                 np.linspace(0, top, 51), "RMS DD final (s)")
    axes[2].set_title("RMS DD après relocalisation")
    save(fig, out / "compare.png")

    map_cfg = config.get("map", {})
    for i, (label, sh) in enumerate(shifts.items()):
        map_and_sections(
            [{"lon": sh["longitude"], "lat": sh["latitude"], "depth": sh["depth"],
              "label": label, "color": SERIES[i]}],
            map_cfg.get("sections"), f"Run {label}", out / f"map_{label}.png",
            zone=map_cfg.get("zone"))
        map_and_sections(
            [{"lon": sh["longitude"], "lat": sh["latitude"], "depth": sh["depth"],
              "label": label, "color": SERIES[i], "time": sh["time"]}],
            map_cfg.get("sections"), f"Run {label} : date des événements",
            out / f"map_{label}_temps.png", zone=map_cfg.get("zone"))

    fig, axes = plt.subplots(len(shifts), 1, figsize=(10, 3 * len(shifts)),
                             sharex=True, sharey=True, squeeze=False)
    for i, (ax, (label, sh)) in enumerate(zip(axes[:, 0], shifts.items())):
        ax.scatter(sh["time"], sh["depth"], s=5, color=SERIES[i], linewidths=0)
        ax.set_ylabel("Profondeur (km)")
        ax.set_title(f"Run {label} (n={len(sh)})")
    axes[0, 0].invert_yaxis()
    axes[-1, 0].set_xlabel("Date (UTC)")
    save(fig, out / "profondeur_temps.png")


if __name__ == "__main__":
    main()
