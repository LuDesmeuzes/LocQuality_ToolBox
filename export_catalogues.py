"""
export_catalogues.py
====================
Exporte les catalogues à comparer au format CSV de l'outil d'analyse de
catalogues dans le navigateur (carte, coupes, 3D, mêmes séismes d'un
catalogue à l'autre, DBSCAN, valeur b…) : un fichier par catalogue, à
glisser dans la page.

Catalogues exportés (section export de config.yaml, sinon section compare) :
  - chaque run NLL (bulletin et runs locaux), avec sa classe de qualité ;
  - chaque run scrtdd, événements relocalisés seulement ;
  - l'origine de référence (LOCSAT) de chaque événement.

L'identifiant evid est l'Event ID SeisComP partout : l'outil apparie ainsi
les mêmes séismes d'un catalogue à l'autre sans passer par le temps origine.
Pour un run scrtdd, il vient de l'eventlist ; sans eventlist, du temps
origine de l'origine de référence (summary.time_tolerance_s).

Colonnes : evid, time (UTC), latitude, longitude, depth_km (sous le niveau
de la mer), erh_km, erz_km, rms_s, puis selon le catalogue :
  - NLL : classe (A/B/C ; la dernière classe de nll.quality_classes est
    laissée vide, l'outil ne connaissant que A, B, C et X), classe_nll (la
    classe complète), nphs, gap, dmin_km ;
  - scrtdd : mag, cluster, rms_dd_debut_s / rms_dd_fin_s (RMS de résidus
    double différence), dh_km / dz_km (déplacement depuis la position de
    départ).

Sorties (dans <output_dir>/export_catalogues/) :
  - <run NLL>.csv, scrtdd_<étiquette>.csv, reference.csv

Utilisation :
    python export_catalogues.py
    python export_catalogues.py --config un_autre_config.yaml
"""

import numpy as np
import pandas as pd

from loc_common import (
    NLL_METHOD, ScrtddRun, classify_quality, load_bulletin, load_config, log, match_by_time,
    nll_origins, output_dir, parse_args, reference_origins, require,
)

def tool_format(df, extra=()):
    """Colonnes de l'outil à partir d'origines au format read_bulletin."""
    out = pd.DataFrame({
        "evid": df["event_id"],
        "time": pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-4],
        "latitude": df["latitude"].round(6),
        "longitude": df["longitude"].round(6),
        "depth_km": df["depth"].round(4),
        "erh_km": df.get("erh_km").round(4) if "erh_km" in df else None,
        "erz_km": df.get("depth_unc_km").round(4) if "depth_unc_km" in df else None,
        "rms_s": df.get("rms").round(4) if "rms" in df else None,
    })
    for col in extra:
        if col in df:
            out[col] = df[col]
    return out


def nll_catalogue(bulletin, author, classes):
    nll = nll_origins(bulletin, author)
    nll["classe_nll"] = classify_quality(nll, classes)
    # L'outil ne connaît que A, B, C (et X) : la classe « reste » est laissée vide
    nll["classe"] = nll["classe_nll"].where(nll["classe_nll"] != classes[-1]["name"], "")
    if "nphs" in nll:
        nll["nphs"] = nll["nphs"].astype("Int64")
    return tool_format(nll, ["classe", "classe_nll", "nphs", "gap", "dmin_km"])


def scrtdd_catalogue(run, ref, tolerance_s):
    sh = run.shifts()
    sh = sh[sh["relocated"].astype(str).str.lower() == "true"].copy()
    ids = run.seiscomp_ids()
    if not ids.empty:
        sh = sh.merge(ids[["id", "event_id"]], on="id", how="left")
    else:
        log(f"Pas d'eventlist : lien par temps origine (± {tolerance_s} s)", 1)
        idx = match_by_time(sh["time"], ref["time"], tolerance_s)
        sh["event_id"] = np.where(idx >= 0, ref["event_id"].to_numpy()[idx], None)
    missing = sh["event_id"].isna()
    if missing.any():
        log(f"⚠ {missing.sum()} événements sans Event ID : identifiant scrtdd gardé", 1)
        sh.loc[missing, "event_id"] = "scrtdd_" + sh.loc[missing, "id"].astype(str)
    sh["cluster"] = sh["id"].map(run.event_clusters())
    sh = sh.rename(columns={"magnitude": "mag", "startRms": "rms_dd_debut_s",
                            "finalRms": "rms_dd_fin_s"})
    sh["dh_km"], sh["dz_km"] = sh["dh_km"].round(4), sh["dz_km"].round(4)
    return tool_format(sh, ["mag", "cluster", "rms_dd_debut_s", "rms_dd_fin_s", "dh_km", "dz_km"])


def write(df, path):
    df.to_csv(path, index=False)
    log(f"{path.name} : {len(df)} séismes", 1)


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    out = output_dir(config, "export_catalogues")
    export = config.get("export") or {}
    compare = config.get("compare") or {}
    classes = require(config, "nll.quality_classes")

    bulletin = load_bulletin(config)
    ref = reference_origins(bulletin)
    if export.get("reference", True):
        write(tool_format(ref), out / "reference.csv")

    authors = export.get("nll_authors") or compare.get("nll_authors") or sorted(
        bulletin.loc[bulletin["method"] == NLL_METHOD, "author"].dropna().unique())
    for author in authors:
        write(nll_catalogue(bulletin, author, classes), out / f"{author}.csv")

    runs = export.get("scrtdd_runs") or compare.get("scrtdd_runs") or {}
    tolerance_s = config.get("summary", {}).get("time_tolerance_s", 1.0)
    for label, run_dir in runs.items():
        log(f"Run scrtdd {label}")
        write(scrtdd_catalogue(ScrtddRun(run_dir), ref, tolerance_s), out / f"scrtdd_{label}.csv")

    log(f"Glisser les fichiers de {out} dans la page de l'outil.")


if __name__ == "__main__":
    main()
