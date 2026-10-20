"""
loc_common.py
=============
Fonctions partagées par tous les scripts d'analyse :
  - lecture de la configuration YAML et interface en ligne de commande
  - lecture des bulletins SeisComP « purifiés » (sortie de purify_cat.py)
  - lecture des sorties scrtdd (eventlist, input-*.csv, reloc-*.csv, cluster-*.csv)
  - conversions géographiques simples (déplacements en km, coupes)
  - style et figures communes (carte + coupes)

Ce module n'a pas vocation à être exécuté directement.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

KM_PER_DEG = 111.195

# Méthode SeisComP des origines NonLinLoc
NLL_METHOD = "NonLinLoc"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION / CLI / LOG
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path):
    """
    Charge un fichier de configuration YAML.

    Si le fichier n'existe pas, lève une erreur explicite invitant
    à copier config.example.yaml → config.yaml et à l'adapter.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        example = config_path.parent / "config.example.yaml"
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {config_path}\n"
            f"→ Copier {example.name} vers {config_path.name} puis adapter "
            f"les chemins à votre machine/cas ."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Fichier de configuration vide ou invalide : {config_path}")
    return config


def parse_args(description):
    """Interface commune à tous les scripts : --config <fichier.yaml>."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "config.yaml"),
        help="Fichier de configuration YAML (défaut : config.yaml à côté des scripts)",
    )
    return parser.parse_args()


def log(msg, indent=0):
    print("  " * indent + msg, flush=True)


def output_dir(config, script_name):
    """Crée et renvoie <paths.output_dir>/<script_name>/."""
    base = Path(config.get("paths", {}).get("output_dir") or "outputs")
    if not base.is_absolute():
        base = Path(__file__).resolve().parent / base
    out = base / script_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def require(config, dotted_key):
    """Renvoie config['a']['b'] pour 'a.b', ou lève une erreur explicite si absent."""
    node = config
    for part in dotted_key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            raise ValueError(f"{dotted_key} doit être renseigné dans config.yaml")
    return node


# ─────────────────────────────────────────────────────────────────────────────
# BULLETIN SEISCOMP (purified_*.csv / coupled_*.csv)
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes de purify_cat.py → noms courts utilisés dans les scripts.
# Les incertitudes SeisComP (latitude, longitude, profondeur) sont en km.
BULLETIN_COLUMNS = {
    "Event ID": "event_id",
    "Origin ID": "origin_id",
    "Origin Time": "time",
    "Latitude": "latitude",
    "Latitude uncertainty": "lat_unc_km",
    "Longitude": "longitude",
    "Longitude uncertainty": "lon_unc_km",
    "Depth": "depth",
    "Depth uncertainty": "depth_unc_km",
    "Event Type": "event_type",
    "Author": "author",
    "Method ID": "method",
    "Preferred Origin ID": "preferred_origin_id",
    "Quality Standard Error": "rms",
    "Distance (km)": "distance_km",
}

NUMERIC_COLUMNS = ["latitude", "lat_unc_km", "longitude", "lon_unc_km", "depth",
                   "depth_unc_km", "rms", "distance_km"]


def read_bulletin(path):
    """
    Lit un bulletin SeisComP purifié (une ligne par origine).

    Les colonnes connues sont renommées (voir BULLETIN_COLUMNS) ; les autres
    sont conservées telles quelles, ce qui permet d'ajouter plus tard des
    colonnes à l'export scquery (gap, nombre de phases…) sans toucher au code.
    Colonne dérivée : erh_km = max(lat_unc_km, lon_unc_km).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bulletin introuvable : {path}")
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=BULLETIN_COLUMNS)
    for col in df.columns:
        df[col] = df[col].str.strip()
    for col in NUMERIC_COLUMNS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if {"lat_unc_km", "lon_unc_km"} <= set(df.columns):
        df["erh_km"] = df[["lat_unc_km", "lon_unc_km"]].max(axis=1)
    return df


def nll_origins(bulletin, author=None):
    """
    Origines NonLinLoc du bulletin, éventuellement filtrées sur un run
    (colonne Author, ex. '1_ubaye_topo_v2'). Une origine par événement :
    s'il y en a plusieurs, la plus récente (Origin ID le plus grand) est gardée.
    """
    df = bulletin[bulletin["method"] == NLL_METHOD]
    if author is not None:
        df = df[df["author"] == author]
        if df.empty:
            authors = sorted(bulletin.loc[bulletin["method"] == NLL_METHOD, "author"].dropna().unique())
            raise ValueError(f"Aucune origine NLL pour l'auteur '{author}'. "
                             f"Auteurs NLL disponibles : {authors}")
    return (df.sort_values("origin_id")
              .drop_duplicates(["event_id", "author"], keep="last")
              .reset_index(drop=True))


def reference_origins(bulletin):
    """
    Origine de référence de chaque événement : l'origine préférée, quand elle
    n'est pas une origine NLL (en pratique la localisation manuelle LOCSAT).
    """
    ref = bulletin[(bulletin["origin_id"] == bulletin["preferred_origin_id"])
                   & (bulletin["method"] != NLL_METHOD)]
    return ref.drop_duplicates("event_id").reset_index(drop=True)


def pair_with_reference(nll, ref):
    """
    Joint origines NLL et origines de référence sur event_id et calcule les
    écarts NLL − référence : de_km, dn_km (est, nord), dh_km (horizontal),
    dz_km (profondeur, positif = NLL plus profond), d3_km, dt_s.
    """
    keep = ["event_id", "time", "latitude", "longitude", "depth", "rms",
            "erh_km", "depth_unc_km", "author", "method"]
    keep = [c for c in keep if c in ref.columns]
    pairs = nll.merge(ref[keep], on="event_id", suffixes=("", "_ref"))
    de, dn = horizontal_shift_km(pairs["longitude_ref"], pairs["latitude_ref"],
                                 pairs["longitude"], pairs["latitude"])
    pairs["de_km"], pairs["dn_km"] = de, dn
    pairs["dh_km"] = np.hypot(de, dn)
    pairs["dz_km"] = pairs["depth"] - pairs["depth_ref"]
    pairs["d3_km"] = np.hypot(pairs["dh_km"], pairs["dz_km"])
    pairs["dt_s"] = (pairs["time"] - pairs["time_ref"]).dt.total_seconds()
    return pairs


def classify_quality(df, classes):
    """
    Attribue une classe de qualité à chaque origine.

    classes : liste ordonnée (meilleure d'abord) de dicts {name, <critères>}.
    Critères reconnus (tous optionnels) :
        rms_max, erh_max_km, erz_max_km      → toujours disponibles
        gap_max, nphs_min, dmin_max_km       → si les colonnes gap / nphs /
                                               dmin_km existent dans le bulletin
    Un critère dont la colonne est absente est ignoré (avec un message).
    Une origine reçoit la première classe dont elle satisfait tous les
    critères, sinon le nom de la dernière classe.
    """
    checks = {
        "rms_max": ("rms", np.less_equal),
        "erh_max_km": ("erh_km", np.less_equal),
        "erz_max_km": ("depth_unc_km", np.less_equal),
        "gap_max": ("gap", np.less_equal),
        "nphs_min": ("nphs", np.greater_equal),
        "dmin_max_km": ("dmin_km", np.less_equal),
    }
    missing = set()
    result = pd.Series(classes[-1]["name"], index=df.index, dtype=object)
    assigned = pd.Series(False, index=df.index)
    for cls in classes:
        ok = pd.Series(True, index=df.index)
        for crit, (col, op) in checks.items():
            if cls.get(crit) is None:
                continue
            if col not in df:
                missing.add(f"{crit} (colonne '{col}')")
                continue
            ok &= op(df[col], cls[crit]).fillna(False)
        result[ok & ~assigned] = cls["name"]
        assigned |= ok
    for m in sorted(missing):
        log(f"⚠ critère ignoré, absent du bulletin : {m}", 1)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SCRTDD : lecture d'un dossier de run
# ─────────────────────────────────────────────────────────────────────────────

class ScrtddRun:
    """
    Accès aux fichiers d'un run scrtdd multi-événements.

    run_dir est le dossier qui contient reloc-event.csv. Selon la façon dont
    scrtdd a été lancé, les fichiers d'entrée et par cluster sont soit dans ce
    même dossier, soit dans un sous-dossier multievent_<date>_<pid>/ : les
    deux cas sont gérés.

    L'id des événements scrtdd est le numéro de ligne (à partir de 1) dans la
    liste d'origines passée à scrtdd (eventlist_*.csv produit par sclistorg) :
    c'est ce qui permet de retrouver l'Event ID SeisComP.
    """

    def __init__(self, run_dir, eventlist=None):
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "reloc-event.csv").exists():
            raise FileNotFoundError(f"reloc-event.csv introuvable dans {self.run_dir}")
        self.work_dir = self._find_work_dir()
        self.eventlist_path = Path(eventlist) if eventlist else self._find_eventlist()

    def _find_work_dir(self):
        if (self.run_dir / "input-event.csv").exists():
            return self.run_dir
        for d in sorted(self.run_dir.glob("multievent_*"), reverse=True):
            if (d / "input-event.csv").exists():
                return d
        return self.run_dir

    def _find_eventlist(self):
        found = sorted(self.run_dir.glob("eventlist*.csv"))
        return found[0] if found else None

    @staticmethod
    def _read(path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _with_time(df):
        if not df.empty:
            df["time"] = pd.to_datetime(df["isotime"], utc=True, format="ISO8601")
        return df

    def input_events(self):
        return self._with_time(self._read(self.work_dir / "input-event.csv"))

    def reloc_events(self):
        return self._with_time(self._read(self.run_dir / "reloc-event.csv"))

    def reloc_phases(self):
        return self._read(self.run_dir / "reloc-phase.csv")

    def stations(self):
        df = self._read(self.work_dir / "input-station.csv")
        return df if not df.empty else self._read(self.run_dir / "reloc-station.csv")

    def xcorr(self):
        return self._read(self.work_dir / "xcorr.csv")

    def cluster_files(self, kind):
        """Fichiers cluster-<N>-<kind>.csv triés par N (kind : 'event', 'pair', ...)."""
        pattern = re.compile(rf"^cluster-(\d+)-{re.escape(kind)}\.csv$")
        files = [(int(m.group(1)), p) for p in self.work_dir.glob("cluster-*.csv")
                 if (m := pattern.match(p.name))]
        return [p for _, p in sorted(files)]

    def double_differences(self, kind="final"):
        """Concatène les cluster-N-<kind>-double-difference.csv avec une colonne 'cluster'."""
        frames = []
        for f in self.cluster_files(f"{kind}-double-difference"):
            df = pd.read_csv(f)
            df["cluster"] = int(f.name.split("-")[1])
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def event_clusters(self):
        """Série id → numéro de cluster (d'après cluster-N-event.csv)."""
        rows = [pd.Series(int(f.name.split("-")[1]), index=pd.read_csv(f, usecols=["id"])["id"])
                for f in self.cluster_files("event")]
        return pd.concat(rows) if rows else pd.Series(dtype=int)

    def seiscomp_ids(self):
        """
        DataFrame id → event_id / origin_id SeisComP, lu dans l'eventlist.
        Vide si l'eventlist est introuvable.
        """
        if self.eventlist_path is None or not self.eventlist_path.exists():
            return pd.DataFrame(columns=["id", "event_id", "origin_id"])
        el = pd.read_csv(self.eventlist_path)
        return pd.DataFrame({"id": np.arange(1, len(el) + 1),
                             "event_id": el["event"].str.strip(),
                             "origin_id": el["origin"].str.strip()})

    def shifts(self):
        """
        Événements relocalisés avec leur position initiale et le déplacement :
        de_km, dn_km, dh_km, dz_km (positif = plus profond après relocalisation).
        """
        inp = self.input_events()
        out = self.reloc_events()
        df = out.merge(inp[["id", "latitude", "longitude", "depth"]],
                       on="id", suffixes=("", "_init"))
        de, dn = horizontal_shift_km(df["longitude_init"], df["latitude_init"],
                                     df["longitude"], df["latitude"])
        df["de_km"], df["dn_km"] = de, dn
        df["dh_km"] = np.hypot(de, dn)
        df["dz_km"] = df["depth"] - df["depth_init"]
        return df


# ─────────────────────────────────────────────────────────────────────────────
# GÉOGRAPHIE
# ─────────────────────────────────────────────────────────────────────────────

def lonlat_to_km(lon, lat, lon0, lat0):
    """Projection équirectangulaire locale (suffisante à l'échelle d'une zone de ~100 km)."""
    x = (np.asarray(lon, dtype=float) - lon0) * KM_PER_DEG * np.cos(np.radians(lat0))
    y = (np.asarray(lat, dtype=float) - lat0) * KM_PER_DEG
    return x, y


def horizontal_shift_km(lon1, lat1, lon2, lat2):
    """Composantes (est, nord) en km du déplacement du point 1 au point 2."""
    lat1, lat2 = np.asarray(lat1, dtype=float), np.asarray(lat2, dtype=float)
    lat_mid = (lat1 + lat2) / 2
    de = (np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)) \
        * KM_PER_DEG * np.cos(np.radians(lat_mid))
    dn = (lat2 - lat1) * KM_PER_DEG
    return de, dn


def project_on_section(lon, lat, section):
    """
    Projette des points sur un profil de coupe.

    section : dict {lon1, lat1, lon2, lat2, half_width_km}
    Renvoie (distance le long du profil en km, masque des points situés à
    moins de half_width_km du profil et entre ses deux extrémités).
    """
    lat0 = (section["lat1"] + section["lat2"]) / 2
    x, y = lonlat_to_km(lon, lat, section["lon1"], lat0)
    x2, y2 = lonlat_to_km(section["lon2"], section["lat2"], section["lon1"], lat0)
    _, y1 = lonlat_to_km(section["lon1"], section["lat1"], section["lon1"], lat0)
    length = np.hypot(x2, y2 - y1)
    ux, uy = x2 / length, (y2 - y1) / length
    along = x * ux + (y - y1) * uy
    perp = -x * uy + (y - y1) * ux
    mask = (np.abs(perp) <= section.get("half_width_km", 2.0)) & (along >= 0) & (along <= length)
    return along, mask


def match_by_time(left_times, right_times, tolerance_s):
    """
    Associe chaque temps de left au temps le plus proche de right (à
    tolerance_s près). Renvoie les indices positionnels dans right
    (-1 si pas de correspondance).
    """
    left = pd.Series(pd.to_datetime(left_times, utc=True)).reset_index(drop=True)
    right = pd.Series(pd.to_datetime(right_times, utc=True)).reset_index(drop=True)
    l_df = pd.DataFrame({"t": left, "li": np.arange(len(left))}).sort_values("t")
    r_df = pd.DataFrame({"t": right, "ri": np.arange(len(right))}).sort_values("t")
    merged = pd.merge_asof(l_df, r_df, on="t", direction="nearest",
                           tolerance=pd.Timedelta(seconds=tolerance_s))
    out = np.full(len(left), -1, dtype=int)
    ok = merged["ri"].notna()
    out[merged.loc[ok, "li"].to_numpy()] = merged.loc[ok, "ri"].astype(int).to_numpy()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

# Palette catégorielle (ordre fixe, jamais cyclé) et rampe séquentielle bleue
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def setup_style():
    """Style matplotlib commun : axes discrets, grille légère, traits fins."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    log(f"→ {path}", 1)


def hist_compare(ax, series_by_label, bins, xlabel):
    """Histogrammes superposés (contours) de plusieurs séries, avec médianes en légende."""
    for i, (label, values) in enumerate(series_by_label.items()):
        values = pd.Series(values).dropna()
        ax.hist(values, bins=bins, histtype="step", linewidth=1.5, color=SERIES[i],
                label=f"{label} (médiane {values.median():.3g}, n={len(values)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Nombre")
    ax.legend(loc="upper right")


def map_and_sections(layers, sections, title, path, vectors=None, stations=None,
                     zone=None):
    """
    Carte + une coupe en profondeur par profil de config.sections.

    layers   : liste de dicts {lon, lat, depth, label, color, [size]}
    vectors  : dict optionnel {lon0, lat0, depth0, lon1, lat1, depth1} pour
               tracer les déplacements (point 0 → point 1)
    stations : DataFrame optionnel avec colonnes longitude, latitude (+ code)
    zone     : dict optionnel {lon_min, lon_max, lat_min, lat_max}
    """
    sections = sections or []
    n = 1 + len(sections)
    fig, axes = plt.subplots(1, n, figsize=(6 + 5 * len(sections), 6),
                             gridspec_kw={"width_ratios": [1.2] + [1] * len(sections)})
    axes = np.atleast_1d(axes)
    ax = axes[0]

    all_lat = np.concatenate([np.asarray(l["lat"], dtype=float) for l in layers])
    lat0 = np.nanmean(all_lat) if len(all_lat) else 45.0

    if vectors is not None:
        ax.plot(np.vstack([vectors["lon0"], vectors["lon1"]]),
                np.vstack([vectors["lat0"], vectors["lat1"]]),
                color=INK_MUTED, linewidth=0.4, alpha=0.6, zorder=1)
    for l in layers:
        ax.scatter(l["lon"], l["lat"], s=l.get("size", 4), color=l["color"],
                   label=f"{l['label']} (n={len(l['lon'])})", linewidths=0, zorder=2)
    if stations is not None and not stations.empty:
        ax.scatter(stations["longitude"], stations["latitude"], marker="^", s=40,
                   facecolor="none", edgecolor=INK, linewidths=1, label="Stations", zorder=3)
    for i, sec in enumerate(sections):
        ax.plot([sec["lon1"], sec["lon2"]], [sec["lat1"], sec["lat2"]],
                color=INK, linewidth=1, zorder=4)
        ax.annotate(sec.get("name", f"S{i + 1}"), (sec["lon1"], sec["lat1"]),
                    fontsize=8, color=INK, xytext=(3, 3), textcoords="offset points")
    if zone:
        ax.set_xlim(zone["lon_min"], zone["lon_max"])
        ax.set_ylim(zone["lat_min"], zone["lat_max"])
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    legend = ax.legend(loc="best")
    for handle in legend.legend_handles:
        handle.set_sizes([25])

    for ax, sec in zip(axes[1:], sections):
        if vectors is not None:
            a0, m0 = project_on_section(vectors["lon0"], vectors["lat0"], sec)
            a1, m1 = project_on_section(vectors["lon1"], vectors["lat1"], sec)
            m = m0 | m1
            ax.plot(np.vstack([a0[m], a1[m]]),
                    np.vstack([np.asarray(vectors["depth0"])[m], np.asarray(vectors["depth1"])[m]]),
                    color=INK_MUTED, linewidth=0.4, alpha=0.6, zorder=1)
        for l in layers:
            along, m = project_on_section(l["lon"], l["lat"], sec)
            ax.scatter(along[m], np.asarray(l["depth"])[m], s=l.get("size", 4),
                       color=l["color"], linewidths=0, zorder=2)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Distance le long du profil (km)")
        ax.set_ylabel("Profondeur (km)")
        ax.set_title(f"Coupe {sec.get('name', '')} (±{sec.get('half_width_km', 2.0)} km)")

    save(fig, path)
