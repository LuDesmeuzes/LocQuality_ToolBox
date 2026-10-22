"""
nll_coupe_interactive.py
========================
Carte et coupe interactives des hypocentres, dans une fenêtre matplotlib
(même principe que coupe_interactive.py de projection_loc).

  - à gauche, la carte des épicentres, colorés par profondeur ;
  - cliquer-glisser sur la carte trace un profil, la coupe s'affiche à droite
    (événements à moins de la demi-largeur choisie du profil) ;
  - boutons : le run affiché (runs de compare.nll_authors, ou l'origine de
    référence LOCSAT ; au départ, nll.author) et « A+B seulement » (événements classés dans
    compare.well_constrained_classes par chacun des runs) ;
  - curseur : demi-largeur de la bande projetée sur la coupe ;
  - cliquer sur un séisme de la coupe affiche ses caractéristiques.

Le profil de départ est le premier de map.sections.

Sorties : aucune (fenêtre interactive). La barre d'outils de matplotlib
permet d'enregistrer la figure.

Utilisation :
    python nll_coupe_interactive.py
    python nll_coupe_interactive.py --config un_autre_config.yaml
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

from loc_common import (
    INK, INK_MUTED, SEQ_BLUE, classify_quality, load_bulletin, load_config, log,
    nll_origins, parse_args, project_on_section, quality_subsets, reference_origins,
    require, setup_style,
)

REFERENCE = "LOCSAT (référence)"

# Profondeurs couvertes par l'échelle de couleur (km)
DEPTH_RANGE = (-2.0, 10.0)


def load_runs(config):
    """{nom affiché: origines}, et l'ensemble des événements A+B de tous les runs."""
    bulletin = load_bulletin(config)
    authors = require(config, "compare.nll_authors")
    runs = {a: nll_origins(bulletin, a) for a in authors}
    classes = require(config, "nll.quality_classes")
    good = config["compare"].get("well_constrained_classes") or ["A", "B"]
    subsets = quality_subsets(runs, classes, good)
    for a, df in runs.items():
        df["quality"] = classify_quality(df, classes).values
    runs[REFERENCE] = reference_origins(bulletin)
    return runs, subsets["_".join(good)], "+".join(good)


class Viewer:
    def __init__(self, runs, good_ids, good_label, zone, section, first=None):
        self.runs, self.good_ids = runs, good_ids
        self.current = first if first in runs else next(iter(runs))
        self.only_good = False
        self.section = dict(section)
        self.cmap = LinearSegmentedColormap.from_list("prof", SEQ_BLUE)
        self.press = None

        self.fig = plt.figure(figsize=(15, 7))
        self.ax_map = self.fig.add_axes([0.05, 0.22, 0.38, 0.70])
        self.ax_sec = self.fig.add_axes([0.50, 0.22, 0.47, 0.70])
        if zone:
            self.ax_map.set_xlim(zone["lon_min"], zone["lon_max"])
            self.ax_map.set_ylim(zone["lat_min"], zone["lat_max"])
        self.ax_map.set_aspect(1 / np.cos(np.radians(self.ax_map.get_ylim()[0])))
        self.ax_map.set_xlabel("Longitude (°)")
        self.ax_map.set_ylabel("Latitude (°)")
        self.scatter = self.ax_map.scatter([], [], c=[], s=4, cmap=self.cmap,
                                           vmin=DEPTH_RANGE[0], vmax=DEPTH_RANGE[1])
        self.fig.colorbar(self.scatter, ax=self.ax_map, label="Profondeur (km)", shrink=0.8)
        self.line, = self.ax_map.plot([], [], color=INK, lw=1.5)
        self.labels = [self.ax_map.text(0, 0, t, color=INK, fontsize=10, fontweight="bold")
                       for t in ("A", "B")]

        # Noms de runs raccourcis pour les boutons
        self.radio_names = {n.split("_20")[0] if n != REFERENCE else n: n for n in runs}
        self.radio = RadioButtons(self.fig.add_axes([0.05, 0.02, 0.30, 0.13]),
                                  list(self.radio_names),
                                  active=list(self.radio_names.values()).index(self.current))
        self.radio.on_clicked(self.on_run)
        self.check = CheckButtons(self.fig.add_axes([0.38, 0.06, 0.12, 0.05]),
                                  [f"{good_label} seulement"], [False])
        self.check.on_clicked(self.on_good)
        self.slider = Slider(self.fig.add_axes([0.58, 0.08, 0.30, 0.03]), "Demi-largeur (km)",
                             0.5, 5.0, valinit=self.section.get("half_width_km", 2.0), valstep=0.5)
        self.slider.on_changed(self.on_width)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.redraw()

    # ── Données ──────────────────────────────────────────────────────────
    def data(self):
        df = self.runs[self.current]
        if self.only_good:
            df = df[df["event_id"].isin(self.good_ids)]
        return df.dropna(subset=["longitude", "latitude", "depth"])

    # ── Dessin ───────────────────────────────────────────────────────────
    def redraw(self):
        df = self.data()
        self.scatter.set_offsets(np.column_stack([df["longitude"], df["latitude"]]))
        self.scatter.set_array(df["depth"].to_numpy())
        self.ax_map.set_title(f"{len(df)} événements (cliquer-glisser : profil)")
        self.draw_section()

    def draw_section(self):
        s = self.section
        self.line.set_data([s["lon1"], s["lon2"]], [s["lat1"], s["lat2"]])
        for label, lon, lat in zip(self.labels, (s["lon1"], s["lon2"]), (s["lat1"], s["lat2"])):
            label.set_position((lon, lat))

        df = self.data()
        along, mask = project_on_section(df["longitude"], df["latitude"], s)
        self.sec_df = df[mask].assign(along=np.asarray(along)[mask])
        ax = self.ax_sec
        ax.cla()
        ax.scatter(self.sec_df["along"], self.sec_df["depth"], c=self.sec_df["depth"], s=6,
                   cmap=self.cmap, vmin=DEPTH_RANGE[0], vmax=DEPTH_RANGE[1], picker=3)
        ax.invert_yaxis()
        ax.set_xlabel("Distance le long du profil A → B (km)")
        ax.set_ylabel("Profondeur (km)")
        ax.set_title(f"Coupe : {len(self.sec_df)} événements à ±{s['half_width_km']:g} km du profil")
        ax.text(1, -0.09, "cliquer sur un séisme pour l'identifier", transform=ax.transAxes,
                ha="right", fontsize=8, color=INK_MUTED)
        self.fig.canvas.draw_idle()

    # ── Événements ───────────────────────────────────────────────────────
    def on_run(self, label):
        self.current = self.radio_names[label]
        self.redraw()

    def on_good(self, _):
        self.only_good = self.check.get_status()[0]
        self.redraw()

    def on_width(self, value):
        self.section["half_width_km"] = value
        self.draw_section()

    def on_press(self, event):
        if event.inaxes is self.ax_map and event.button == 1 \
                and self.fig.canvas.toolbar.mode == "":
            self.press = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.press is None or event.inaxes is not self.ax_map:
            return
        self.line.set_data([self.press[0], event.xdata], [self.press[1], event.ydata])
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.press is None:
            return
        start, self.press = self.press, None
        if event.inaxes is not self.ax_map or (start[0], start[1]) == (event.xdata, event.ydata):
            self.draw_section()
            return
        self.section.update(lon1=start[0], lat1=start[1], lon2=event.xdata, lat2=event.ydata)
        self.draw_section()

    def on_pick(self, event):
        if event.artist.axes is not self.ax_sec or not len(event.ind):
            return
        ev = self.sec_df.iloc[event.ind[0]]
        info = f"{ev['event_id']}  {ev['time']:%Y-%m-%d %H:%M:%S}  prof. {ev['depth']:.2f} km"
        for key, fmt in (("rms", "RMS {:.3f} s"), ("erh_km", "erh {:.2f} km"),
                         ("depth_unc_km", "erz {:.2f} km"), ("quality", "classe {}")):
            if key in ev and ev[key] == ev[key]:
                info += "  " + fmt.format(ev[key])
        log(info)
        self.ax_sec.set_title(info, fontsize=9)
        self.fig.canvas.draw_idle()


def main():
    args = parse_args(__doc__.splitlines()[1])
    config = load_config(args.config)
    # loc_common impose Agg (figures enregistrées) : retour au backend interactif par défaut
    plt.switch_backend(matplotlib.rcsetup._auto_backend_sentinel)
    setup_style()
    runs, good_ids, good_label = load_runs(config)
    map_cfg = config.get("map") or {}
    sections = map_cfg.get("sections") or []
    zone = map_cfg.get("zone")
    if sections:
        section = sections[0]
    else:
        lon, lat = runs[next(iter(runs))][["longitude", "latitude"]].median()
        section = {"lon1": lon - 0.1, "lat1": lat, "lon2": lon + 0.1, "lat2": lat}
    section.setdefault("half_width_km", 2.0)
    viewer = Viewer(runs, good_ids, good_label, zone, section,
                    first=(config.get("nll") or {}).get("author"))
    plt.show()


if __name__ == "__main__":
    main()
