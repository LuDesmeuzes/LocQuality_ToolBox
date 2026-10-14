"""
summary_table.py
================
Tableau récapitulatif par événement, qui rassemble :
  - la localisation NLL du run nll.author (bulletin SeisComP) et sa classe
    de qualité
  - l'écart NLL − origine de référence (LOCSAT)
  - le résultat scrtdd : relocalisé ou non, cluster, position relocalisée,
    déplacement, RMS DD avant / après, nombre de doubles différences

Le lien bulletin ↔ scrtdd se fait par l'Event ID SeisComP, retrouvé grâce à
l'eventlist passée à scrtdd (l'id scrtdd est le numéro de ligne dans ce
fichier). Sans eventlist, il se fait par temps origine
(summary.time_tolerance_s).

Lancer dd_links.py avant pour avoir le nombre de doubles différences par
événement (lu dans outputs/dd_links/event_links.csv, optionnel).

Sorties (dans <output_dir>/summary_table/) :
  - summary.csv : une ligne par événement du bulletin NLL
  - stats.csv   : taux de relocalisation et déplacements par classe NLL

Utilisation :
    python summary_table.py
    python summary_table.py --config un_autre_config.yaml
"""

import numpy as np
import pandas as pd

from loc_common import (
    ScrtddRun, classify_quality, load_config, log, match_by_time, nll_origins,
    output_dir, pair_with_reference, parse_args, read_bulletin,
    reference_origins, require,
)


def scrtdd_table(run, tolerance_s, nll):
    """Résultats scrtdd par événement, avec la colonne event_id SeisComP."""
    inp = run.input_events()[["id", "time"]]
    sh = run.shifts()
    cols = {"latitude": "dd_latitude", "longitude": "dd_longitude", "depth": "dd_depth",
            "dh_km": "dd_dh_km", "dz_km": "dd_dz_km",
            "startRms": "dd_rms_start", "finalRms": "dd_rms_final"}
    dd = inp.merge(sh[["id", *cols]].rename(columns=cols), on="id", how="left")
    dd["dd_relocated"] = dd["dd_latitude"].notna()
    dd["dd_cluster"] = dd["id"].map(run.event_clusters())

    ids = run.seiscomp_ids()
    if not ids.empty:
        log(f"Lien par Event ID via {run.eventlist_path.name}", 1)
        dd = dd.merge(ids[["id", "event_id"]], on="id", how="left")
    else:
        log(f"Pas d'eventlist : lien par temps origine (± {tolerance_s} s)", 1)
        idx = match_by_time(dd["time"], nll["time"], tolerance_s)
        dd["event_id"] = np.where(idx >= 0, nll["event_id"].to_numpy()[idx], None)
    return dd.rename(columns={"id": "dd_id"}).drop(columns="time")


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    out = output_dir(config, "summary_table")
    author = require(config, "nll.author")

    log(f"Lecture du bulletin {require(config, 'paths.bulletin')}")
    bulletin = read_bulletin(config["paths"]["bulletin"])
    nll = nll_origins(bulletin, author)
    nll["quality"] = classify_quality(nll, require(config, "nll.quality_classes"))
    pairs = pair_with_reference(nll, reference_origins(bulletin))
    ref_cols = ["event_id", "latitude_ref", "longitude_ref", "depth_ref",
                "method_ref", "dh_km", "dz_km", "d3_km"]
    summary = nll.merge(pairs[ref_cols].rename(columns={"dh_km": "ref_dh_km",
                                                        "dz_km": "ref_dz_km",
                                                        "d3_km": "ref_d3_km"}),
                        on="event_id", how="left")

    run_dir = config["paths"].get("scrtdd_run_dir")
    if run_dir:
        log(f"Lecture du run scrtdd {run_dir}")
        run = ScrtddRun(run_dir, config["paths"].get("scrtdd_eventlist"))
        dd = scrtdd_table(run, config.get("summary", {}).get("time_tolerance_s", 1.0), nll)
        links = output_dir(config, "dd_links") / "event_links.csv"
        if links.exists():
            l = pd.read_csv(links)[["id", "n_dd", "n_neighbours"]]
            dd = dd.merge(l.rename(columns={"id": "dd_id", "n_dd": "dd_n_dd",
                                            "n_neighbours": "dd_n_neighbours"}),
                          on="dd_id", how="left")
        else:
            log("event_links.csv absent : lancer dd_links.py pour avoir le nombre de DD", 1)
        matched = dd["event_id"].isin(summary["event_id"])
        log(f"{matched.sum()} / {len(dd)} événements scrtdd retrouvés dans le bulletin NLL", 1)
        summary = summary.merge(dd[matched], on="event_id", how="left")
        summary["dd_in_input"] = summary["dd_id"].notna()
        summary["dd_relocated"] = summary["dd_relocated"].astype("boolean").fillna(False)

    summary.to_csv(out / "summary.csv", index=False)
    log(f"→ {out / 'summary.csv'} ({len(summary)} événements)", 1)

    if "dd_in_input" in summary:
        sub = summary[summary["dd_in_input"]]
        stats = sub.groupby("quality").agg(
            n_scrtdd_input=("event_id", "size"),
            pct_relocated=("dd_relocated", lambda s: 100 * s.mean()),
            dd_dh_median_km=("dd_dh_km", "median"),
            dd_dz_median_km=("dd_dz_km", "median"),
            ref_d3_median_km=("ref_d3_km", "median"))
        log(stats.round(2).to_string(), 1)
        stats.to_csv(out / "stats.csv")


if __name__ == "__main__":
    main()
