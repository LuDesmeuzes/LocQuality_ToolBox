"""
nll_local_vs_seiscomp.py
========================
Compare un run NLL local (nll_run.py) aux localisations NonLinLoc faites
par SeisComP sur les mêmes origines et les mêmes pointés (screloc --ep sur
les XML journaliers du cache local, préparés par profil_vers_seiscomp.py de
Loc_ToolBox). Avec le même profil et les mêmes grilles des deux côtés,
les positions doivent être identiques : les écarts montrent ce qui diffère
entre le calcul local et SeisComP (version de NLL, phases utilisées,
configuration du plugin).

Pour chaque événement présent des deux côtés : écarts horizontal, vertical
et 3D (m), écart en temps origine, différences de RMS, d'incertitudes et de
nombre de phases, phases utilisées d'un seul côté. Par station : résidus
moyens des deux côtés, sur les phases communes.

Sorties (dans <output_dir>/nll_local_vs_seiscomp/) :
  - pairs.csv          : une ligne par événement (local, SeisComP, écarts)
  - resume.csv         : part des événements identiques à 1 m, 10 m, 100 m,
                         1 km près ; médiane, 95e centile et maximum des écarts
  - stations.csv       : par station, phases des deux côtés ou d'un seul,
                         résidus moyens local / SeisComP et écart moyen
  - ecarts.png         : distribution des écarts 3D, écart en fonction du
                         nombre de phases et de la distance de la station la
                         plus lointaine, événements aux mêmes phases envoyées
                         des deux côtés et autres séparés
  - map_ecarts.png     : carte et coupes, local → SeisComP, événements
                         identiques et différents
  - stations.png       : résidu moyen par station, local contre SeisComP

Configuration (section local_vs_seiscomp de config.yaml) :
  run        : dossier du run local (nll_run.py)
  screloc    : dossier des XML de sortie de screloc
  id_suffix  : suffixe des origines relocalisées (--origin-id-suffix, ".nll")
  tolerance_m : écart 3D en dessous duquel deux positions sont identiques

Utilisation :
    python nll_local_vs_seiscomp.py
    python nll_local_vs_seiscomp.py --config un_autre_config.yaml
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loc_common import (
    INK_MUTED, SERIES, horizontal_shift_km, load_config, log, map_and_sections, output_dir,
    parse_args, read_nll_phases, read_nll_run, read_scml_origins, require, save, setup_style,
)

THRESHOLDS_M = [1, 10, 100, 1000]


def load_screloc(folder, id_suffix):
    """Origines relocalisées (suffixe id_suffix) et leurs arrivées, de tous les XML du dossier."""
    files = sorted(Path(folder).glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier XML dans {folder}")
    origins, arrivals = [], []
    for f in files:
        o, a = read_scml_origins(f, id_suffix=id_suffix)
        o = o[o["origin_id"].str.endswith(id_suffix)]
        origins.append(o)
        arrivals.append(a[a["origin_id"].isin(o["origin_id"])])
    origins = pd.concat(origins, ignore_index=True)
    arrivals = pd.concat(arrivals, ignore_index=True)
    arrivals = arrivals.merge(origins[["origin_id", "event_id"]], on="origin_id")
    log(f"{len(files)} fichiers, {len(origins)} origines relocalisées, {len(arrivals)} arrivées", 1)
    return origins, arrivals


def pair_runs(local, sc):
    """Événements des deux côtés et écarts SeisComP − local (m, s)."""
    keep = ["event_id", "time", "latitude", "longitude", "depth", "rms", "nphs", "gap",
            "lat_unc_km", "lon_unc_km", "depth_unc_km"]
    pairs = local[keep].merge(sc[keep], on="event_id", suffixes=("_local", "_sc"))
    de, dn = horizontal_shift_km(pairs["longitude_local"], pairs["latitude_local"],
                                 pairs["longitude_sc"], pairs["latitude_sc"])
    pairs["de_m"], pairs["dn_m"] = de * 1000, dn * 1000
    pairs["dh_m"] = np.hypot(pairs["de_m"], pairs["dn_m"])
    pairs["dz_m"] = (pairs["depth_sc"] - pairs["depth_local"]) * 1000
    pairs["d3_m"] = np.hypot(pairs["dh_m"], pairs["dz_m"])
    pairs["dt_s"] = (pairs["time_sc"] - pairs["time_local"]).dt.total_seconds()
    pairs["drms_s"] = pairs["rms_sc"] - pairs["rms_local"]
    pairs["dnphs"] = pairs["nphs_sc"] - pairs["nphs_local"]
    for c in ("lat_unc_km", "lon_unc_km", "depth_unc_km"):
        pairs[f"d{c}"] = pairs[f"{c}_sc"] - pairs[f"{c}_local"]
    return pairs


def compare_phases(local_phases, arrivals):
    """
    Phases (événement, station, P ou S) envoyées à NLL des deux côtés :
    local = phases de phases.csv, SeisComP = arrivées timeUsed. Le poids
    final (weight_local, weight_sc) est celui que NLL leur a donné (0 pour
    une phase écartée comme aberrante). Renvoie la table jointe (colonne
    side : both, local, sc).
    """
    key = ["event_id", "station", "phase_type"]
    loc = local_phases.drop_duplicates(key)[key + ["residual_s", "weight", "dist_km"]]
    sc = arrivals[arrivals["time_used"]].drop_duplicates(key)[key + ["residual_s", "weight"]]
    both = loc.merge(sc, on=key, how="outer", suffixes=("_local", "_sc"), indicator=True)
    both["side"] = both["_merge"].map({"both": "both", "left_only": "local", "right_only": "sc"})
    return both.drop(columns="_merge")


def station_table(phases):
    """Par station : phases des deux côtés / local seul / SeisComP seul, résidus moyens."""
    counts = phases.pivot_table(index="station", columns="side", values="event_id",
                                aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=["both", "local", "sc"], fill_value=0)
    counts.columns = ["n_communes", "n_local_seul", "n_seiscomp_seul"]
    common = phases[phases["side"] == "both"].assign(
        dres_s=lambda d: d["residual_s_sc"] - d["residual_s_local"],
        poids_nul_local=lambda d: d["weight_local"] == 0,
        poids_nul_sc=lambda d: d["weight_sc"] == 0)
    res = common.groupby("station").agg(residu_local_s=("residual_s_local", "mean"),
                                        residu_seiscomp_s=("residual_s_sc", "mean"),
                                        ecart_residu_s=("dres_s", "mean"),
                                        n_poids_nul_local=("poids_nul_local", "sum"),
                                        n_poids_nul_seiscomp=("poids_nul_sc", "sum"),
                                        distance_km=("dist_km", "median"))
    return counts.join(res).sort_values("n_communes", ascending=False)


def summary(pairs, n_local, n_sc):
    rows = {"événements locaux": n_local, "événements SeisComP": n_sc,
            "événements communs": len(pairs)}
    for t in THRESHOLDS_M:
        rows[f"identiques à {t} m près (%)"] = round(100 * (pairs["d3_m"] < t).mean(), 2)
    for col, label in [("dh_m", "écart horizontal (m)"), ("dz_m", "écart vertical (m)"),
                       ("d3_m", "écart 3D (m)"), ("dt_s", "écart temps origine (s)"),
                       ("drms_s", "écart RMS (s)")]:
        v = pairs[col].abs()
        rows[f"{label}, médiane"] = round(v.median(), 4)
        rows[f"{label}, 95e centile"] = round(v.quantile(0.95), 4)
        rows[f"{label}, maximum"] = round(v.max(), 4)
    rows["même nombre de phases (%)"] = round(100 * (pairs["dnphs"] == 0).mean(), 2)
    same = pairs[pairs["phases_un_cote"] == 0]
    rows["événements aux mêmes phases envoyées"] = len(same)
    for t in THRESHOLDS_M[1:3]:
        rows[f"  dont identiques à {t} m près (%)"] = round(100 * (same["d3_m"] < t).mean(), 2)
    return pd.Series(rows)


def plot_ecarts(pairs, tolerance_m, path):
    """Écarts 3D, en séparant les événements aux mêmes phases envoyées des autres."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    floor = 0.1
    d3 = pairs["d3_m"].clip(lower=floor)
    bins = np.logspace(np.log10(floor), np.log10(max(d3.max(), 10) * 1.1), 50)
    groups = [("mêmes phases", pairs["phases_un_cote"] == 0),
              ("phases d'un seul côté", pairs["phases_un_cote"] > 0)]

    ax = axes[0]
    for i, (label, m) in enumerate(groups):
        share = 100 * (pairs.loc[m, "d3_m"] < tolerance_m).mean()
        ax.hist(d3[m], bins=bins, histtype="step", linewidth=1.5, color=SERIES[i],
                label=f"{label} (n={m.sum()}, {share:.1f} % < {tolerance_m:g} m)")
    ax.axvline(tolerance_m, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Écart 3D SeisComP − local (m, < {floor:g} m ramenés à {floor:g})")
    ax.set_ylabel("Nombre d'événements")
    ax.set_title(f"Local vs SeisComP (n={len(pairs)})")
    ax.legend(loc="upper right")

    for ax, col, xlabel in [(axes[1], "nphs_local", "Nombre de phases (local)"),
                            (axes[2], "dmax_km", "Distance de la station la plus lointaine (km)")]:
        for i, (label, m) in enumerate(groups):
            ax.scatter(pairs.loc[m, col], d3[m], s=4, color=SERIES[i], alpha=0.5,
                       linewidths=0, label=label)
        ax.axhline(tolerance_m, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Écart 3D (m)")
        ax.legend(loc="upper right")
    save(fig, path)


def plot_stations(stations, path, n_labels=8):
    st = stations.dropna(subset=["residu_local_s", "residu_seiscomp_s"])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(st["residu_local_s"], st["residu_seiscomp_s"], s=12 + 60 * st["n_communes"]
               / max(st["n_communes"].max(), 1), color=SERIES[0], alpha=0.7, linewidths=0)
    lim = float(np.nanmax(np.abs(st[["residu_local_s", "residu_seiscomp_s"]].to_numpy()))) * 1.1 or 0.1
    ax.plot([-lim, lim], [-lim, lim], color=INK_MUTED, linewidth=1, linestyle="--", label="y = x")
    for name, r in st.reindex(st["ecart_residu_s"].abs().sort_values(ascending=False).index)\
            .head(n_labels).iterrows():
        ax.annotate(name, (r["residu_local_s"], r["residu_seiscomp_s"]), fontsize=7,
                    color=INK_MUTED, xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Résidu moyen local (s)")
    ax.set_ylabel("Résidu moyen SeisComP (s)")
    ax.set_title("Résidus par station, phases communes")
    ax.legend(loc="upper left")
    save(fig, path)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    setup_style()
    out = output_dir(config, "nll_local_vs_seiscomp")
    cfg = require(config, "local_vs_seiscomp")
    run_dir = Path(require(config, "local_vs_seiscomp.run"))
    id_suffix = cfg.get("id_suffix", ".nll")
    tolerance_m = float(cfg.get("tolerance_m", 10))

    log(f"Run local {run_dir.name}")
    bulletin = read_nll_run(run_dir)
    local = bulletin[bulletin["author"] != "reference"]
    local_phases = read_nll_phases(run_dir)
    log(f"{len(local)} événements localisés, {len(local_phases)} phases", 1)
    log(f"Sorties screloc {require(config, 'local_vs_seiscomp.screloc')}")
    sc, arrivals = load_screloc(cfg["screloc"], id_suffix)
    sc = sc.dropna(subset=["event_id"]).drop_duplicates("event_id", keep="last")
    log("Modèle SeisComP : " + ", ".join(sc["earth_model"].dropna().unique()), 1)

    pairs = pair_runs(local, sc)
    dmax = local_phases.groupby("event_id")["dist_km"].max().rename("dmax_km")
    pairs = pairs.merge(dmax, on="event_id", how="left")
    phases = compare_phases(local_phases, arrivals)
    diff = phases[phases["side"] != "both"].groupby("event_id").size().rename("phases_un_cote")
    pairs = pairs.merge(diff, on="event_id", how="left").fillna({"phases_un_cote": 0})

    only_local = sorted(set(local["event_id"]) - set(sc["event_id"]))
    only_sc = sorted(set(sc["event_id"]) - set(local["event_id"]))
    log(f"{len(pairs)} événements communs, {len(only_local)} seulement en local, "
        f"{len(only_sc)} seulement dans SeisComP", 1)

    res = summary(pairs, len(local), len(sc))
    log(res.to_string(), 1)
    different = pairs[pairs["d3_m"] >= tolerance_m]
    if len(different):
        share = 100 * (different["phases_un_cote"] > 0).mean()
        log(f"{len(different)} événements à plus de {tolerance_m:g} m : {share:.0f} % ont des "
            "phases utilisées d'un seul côté", 1)

    stations = station_table(phases)
    log("Stations aux phases les plus souvent utilisées d'un seul côté :", 1)
    one_side = stations.assign(n=stations["n_local_seul"] + stations["n_seiscomp_seul"])
    log(one_side[one_side["n"] > 0].sort_values("n", ascending=False).head(10)
        [["n_communes", "n_local_seul", "n_seiscomp_seul", "ecart_residu_s"]].round(4)
        .to_string(), 2)

    pairs.sort_values("d3_m", ascending=False).to_csv(out / "pairs.csv", index=False)
    res.to_csv(out / "resume.csv", header=["valeur"])
    stations.to_csv(out / "stations.csv")
    log(f"→ {out / 'pairs.csv'}", 1)

    log("Figures")
    plot_ecarts(pairs, tolerance_m, out / "ecarts.png")
    plot_stations(stations, out / "stations.png")
    same = pairs[pairs["d3_m"] < tolerance_m]
    map_cfg = config.get("map", {})
    map_and_sections(
        [{"lon": same["longitude_local"], "lat": same["latitude_local"],
          "depth": same["depth_local"], "label": f"identiques (< {tolerance_m:g} m)",
          "color": SERIES[0]},
         {"lon": different["longitude_local"], "lat": different["latitude_local"],
          "depth": different["depth_local"], "label": "local (différents)", "color": SERIES[1],
          "size": 8},
         {"lon": different["longitude_sc"], "lat": different["latitude_sc"],
          "depth": different["depth_sc"], "label": "SeisComP (différents)", "color": SERIES[2],
          "size": 8}],
        map_cfg.get("sections"), "Local → SeisComP", out / "map_ecarts.png",
        vectors={"lon0": different["longitude_local"], "lat0": different["latitude_local"],
                 "depth0": different["depth_local"], "lon1": different["longitude_sc"],
                 "lat1": different["latitude_sc"], "depth1": different["depth_sc"]},
        zone=map_cfg.get("zone"))


if __name__ == "__main__":
    main()
