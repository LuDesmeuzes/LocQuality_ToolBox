# Analyse de la qualité des localisations NLL et des relocalisations scrtdd

Ce dépôt contient des scripts Python pour mesurer la qualité :
- des **localisations NonLinLoc (NLL)** faites dans SeisComP, à partir des bulletins exportés avec scquery puis purifiés (`purify_cat.py`) ;
- des **relocalisations par double différence** faites avec **scrtdd** (hypoDD dans SeisComP), à partir des fichiers CSV écrits par scrtdd en mode multi-événements.

Chaque script permet d'appeler un type d'analyse (décrite ci-dessous).
Le code commun (lecture des fichiers, calculs géographiques, style des figures) est dans `loc_common.py`.

## Scripts disponibles

| Script | Données | Ce qu'il mesure |
|---|---|---|
| `nll_quality.py` | bulletin | Classes de qualité A/B/C/D des origines NLL d'un run |
| `nll_vs_reference.py` | bulletin | Écarts NLL − origine de référence (LOCSAT) |
| `nll_compare_runs.py` | bulletin | Comparaison de plusieurs runs NLL |
| `nll_gridding.py` | bulletin | Quadrillage (octree) et paliers de profondeur des runs NLL |
| `nll_station_residuals.py` | runs locaux | Résidus par station (biais, dispersion) de plusieurs runs |
| `nll_sections.py` | bulletin, runs locaux | Netteté des structures : cartes et coupes côte à côte |
| `nll_coupe_interactive.py` | bulletin, runs locaux | Carte et coupe interactives (profil tracé à la souris) |
| `dd_shifts.py` | scrtdd | Déplacements avant → après relocalisation |
| `dd_residuals.py` | scrtdd | Résidus avant / après, par station, double différence |
| `dd_links.py` | scrtdd | Liens DD par événement, clusters, contrôle de la corrélation croisée |
| `dd_compare_runs.py` | scrtdd | Comparaison de plusieurs runs scrtdd |
| `summary_table.py` | les deux | Tableau récapitulatif par événement |

### `nll_quality.py` — classes de qualité NLL

Lit les origines du run NLL `nll.author` et attribue à chacune une classe
de qualité selon `nll.quality_classes` (RMS, incertitude horizontale,
incertitude en profondeur). Une origine reçoit la première classe dont elle
respecte tous les critères.

Sorties : `nll_quality.csv` (une ligne par origine, avec sa classe),
`nll_quality_stats.csv`, `distributions.png`, `map_classes.png`.

> Le RMS seul est trompeur : un événement localisé avec peu de phases
> peut avoir un RMS très faible. Si l'export scquery contient le gap, le
> nombre de phases ou la distance à la station la plus proche (colonnes
> `gap`, `nphs`, `dmin_km`), les critères `gap_max`, `nphs_min` et
> `dmin_max_km` sont appliqués automatiquement.

### `nll_vs_reference.py` — écart à la référence

Pour chaque événement, compare l'origine NLL du run `nll.author` à
l'origine de référence, c'est-à-dire l'origine préférée non NLL (en
pratique la localisation manuelle LOCSAT). Il calcule l'écart horizontal
(est, nord), l'écart en profondeur (positif = NLL plus profond) et l'écart
3D, globalement et par classe de qualité.

Sorties : `pairs.csv`, `stats.csv`, `shifts.png` (histogrammes et nuage
est-nord, dont la médiane indique un biais systématique), `shift_vs_unc.png`
(écart en fonction de l'incertitude NLL annoncée), `map_shifts.png`.

> La référence n'est pas la vérité : LOCSAT utilise un modèle 1D. Un
> faible écart à la référence veut dire « proche de LOCSAT », pas
> « mieux localisé ».

### `nll_compare_runs.py` — comparaison de runs NLL

Compare les runs NLL listés dans `compare.nll_authors` (ou tous ceux du
bulletin si la liste est vide). Les indicateurs sont calculés pour chaque
run sur tous ses événements, puis sur les seuls événements communs à tous
les runs, pour ne comparer que le même échantillon.

Sorties : `runs.csv`, `compare.png` et `compare_common.png` (un run par
ligne : médiane et intervalle 16–84 % du RMS, de l'incertitude horizontale
et de l'écart à la référence).

### `nll_gridding.py` — quadrillage et paliers de profondeur

NLL place chaque hypocentre au centre de la cellule d'octree la plus
probable. Pour chaque run de `gridding.runs`, le script retrouve, à partir de
`LOCGRID` et `LOCSEARCH OCT` du profil, le niveau de subdivision de la
cellule finale de chaque origine, et donc sa taille : les événements à
cellule grossière sont ceux qui dessinent le quadrillage. Il mesure aussi la
part d'épicentres strictement identiques, les paliers de profondeur
(événements empilés sur une profondeur entière, comparés au voisinage), le
toit de la grille et, si `gridding.topo` est renseigné, les événements
au-dessus du relief.

Sorties : `gridding.csv` (une ligne par origine et par run),
`gridding_stats.csv`, `plateaus.csv`, `cells.png` (répartition des tailles
de cellule finale), `depths.png` (profondeurs au pas de 50 m).

> Les paramètres `locgrid` et `oct` doivent être ceux du profil SeisComP
> qui a produit le run (`NLL.<profil>.conf` sur la machine SeisComP) : avec
> d'autres valeurs, le niveau d'octree n'est pas retrouvé. Les runs locaux
> de `paths.nll_runs` sont ajoutés automatiquement, avec les paramètres de
> leur `run.yaml`.

### `nll_station_residuals.py` — résidus par station

Le résidu d'une phase est le temps observé (pointé) moins le temps calculé
par NLL depuis l'hypocentre. Regroupés par station et par type de phase, les
résidus donnent le **biais** de la station (résidu moyen : le modèle prédit
systématiquement trop tôt ou trop tard pour les trajets vers cette station)
et sa **dispersion**. Un bon modèle donne des biais proches de 0 partout.

Le script compare les runs de `compare.nll_authors` qui ont leurs phases,
c'est-à-dire les runs locaux de `paths.nll_runs` (`phases.csv` écrit par
`nll_run.py`) : l'export du bulletin SeisComP ne contient pas les arrivées.
Les runs sont comparés sur les mêmes événements, pour deux sous-ensembles :
tous les événements communs, puis ceux classés dans
`compare.well_constrained_classes` (ex. A et B) par **chacun** des runs.

Sorties, par sous-ensemble : `stations.csv`, `summary.csv`,
`station_bias.png` (biais par station, un point par run, stations triées par
distance), `residual_vs_distance.png`.

> Une partie de l'erreur du modèle est absorbée par la position de
> l'hypocentre : les résidus la sous-estiment.

### `nll_sections.py` — netteté des structures

Une faille est fine ; les erreurs de localisation étalent la sismicité
autour. Le script trace, pour les runs de `compare.nll_authors` et sur les
mêmes événements, une carte par run et les coupes de `map.sections` (une
ligne par run), et mesure dans chaque coupe la distance médiane au plus
proche voisin et l'épaisseur du nuage (écart-type selon son petit axe), plus
la distance au plus proche voisin en 3D sur toute la zone. Mêmes deux
sous-ensembles que `nll_station_residuals.py`.

Sorties, par sous-ensemble : `metrics.csv`, `maps.png`, `sections.png`.

> Le quadrillage rend la sismicité artificiellement « nette » : la part de
> plus proches voisins confondus (< 10 m) le signale. À lire avec
> `nll_gridding.py`.

### `nll_coupe_interactive.py` — carte et coupe interactives

Ouvre une fenêtre matplotlib : la carte des épicentres à gauche, colorés par
profondeur, et une coupe à droite. Cliquer-glisser sur la carte trace un
nouveau profil. Des boutons choisissent le run (ceux de
`compare.nll_authors`, ou l'origine de référence LOCSAT) et limitent
l'affichage aux événements A+B ; un curseur règle la demi-largeur de la
bande projetée. Cliquer sur un séisme de la coupe affiche son identifiant,
sa profondeur, son RMS, ses incertitudes et sa classe. Le profil de départ
est le premier de `map.sections`.

Sortie : aucune (la barre d'outils permet d'enregistrer la figure).

### `dd_shifts.py` — déplacements de relocalisation

Compare la position initiale de chaque événement (`input-event.csv`,
c'est-à-dire l'origine NLL passée à scrtdd) à sa position relocalisée
(`reloc-event.csv`). La médiane des déplacements est-nord et verticaux
indique une translation d'ensemble du nuage : la double différence
contraint mal la position absolue.

Sorties : `shifts.csv`, `not_relocated.csv`, `shifts_stats.csv`,
`shifts.png`, `map_shifts.png` (vecteurs initial → relocalisé),
`map_before.png`, `map_after.png`.

### `dd_residuals.py` — résidus

- RMS par événement avant / après (`startRms`, `finalRms`). **Ce sont des
  RMS de résidus double différence**, pas des résidus absolus.
- Résidus absolus par phase et par station (`reloc-phase.csv`). Une
  station dont le résidu moyen est loin de 0 est candidate à une correction
  de station ou à une vérification des pointés.
- Résidus double différence finaux en fonction de la distance
  inter-événements, en séparant données catalogue et corrélation croisée.

Sorties : `station_residuals.csv`, `rms.png`, `station_residuals.png`,
`dd_residuals.png`.

### `dd_links.py` — liens et corrélation croisée

- Par événement : nombre de doubles différences utilisées (poids > 0) par
  type (catalogue `ct` / corrélation `cc`) et par phase, nombre de voisins.
- Taille des clusters, événements isolés (hors de tout cluster) et
  événements en cluster mais non relocalisés.
- Corrélation croisée (`xcorr.csv`) : coefficients et décalages (lags).
  Si beaucoup de lags sont au bord de la fenêtre de recherche, ces
  mesures sont probablement fausses. Il faut alors revoir la fenêtre ou le
  seuil de coefficient dans le profil scrtdd.

Sorties : `event_links.csv`, `clusters.csv`, `links.png`, `xcorr.png`.

### `dd_compare_runs.py` — comparaison de runs scrtdd

Compare les runs listés dans `compare.scrtdd_runs` (étiquette → dossier) :
taux de relocalisation, déplacements, RMS DD, résidus absolus, qualité de la
corrélation croisée.

Sorties : `runs.csv`, `compare.png`, `map_<run>.png`, `map_<run>_temps.png`
(mêmes carte et coupes, événements colorés par date) et `profondeur_temps.png`
(profondeur en fonction du temps, un panneau par run) pour suivre une
migration de la sismicité.

Les runs lancés avec `--reloc-catalog` (fichiers de `scrtdd_inputs.py`) sont
lus aussi : positions de départ dans `event.csv`, stations dans `station.csv`.

### `summary_table.py` — tableau récapitulatif

Une ligne par événement du run NLL `nll.author`, avec sa classe de
qualité, son écart à la référence et le résultat scrtdd (relocalisé ou
non, cluster, position relocalisée, déplacement, RMS DD, nombre de DD).
Lancer `dd_links.py` avant pour avoir le nombre de DD.

Le lien bulletin ↔ scrtdd se fait par l'Event ID SeisComP. L'`id` d'un
événement scrtdd est son numéro de ligne dans l'eventlist passée à scrtdd
(sortie de sclistorg). Sans eventlist, le lien se fait par temps origine,
à `summary.time_tolerance_s` près.

Sorties : `summary.csv`, `stats.csv` (taux de relocalisation et déplacements
par classe NLL).

## Données d'entrée

### Bulletin SeisComP (NLL)

Fichier `purified_*.csv` produit par `purify_cat.py` à partir d'un export
scquery : une ligne par **origine**. Chaque événement a donc plusieurs
lignes : son origine manuelle préférée et une origine par run NLL (colonne
`Author`). Les fichiers `coupled_*.csv` (sortie de `make_event_couples.py`)
ont le même format et fonctionnent aussi.

Colonnes utilisées : `Event ID`, `Origin ID`, `Origin Time`, `Latitude`,
`Longitude`, `Depth`, leurs incertitudes (en **km** dans SeisComP),
`Author`, `Method ID`, `Preferred Origin ID`, `Quality Standard Error` (RMS).

> `Origin Time` est tronqué à la seconde par l'export : les écarts en temps
> origine ne sont pas exploitables.

### Run NLL en ligne de commande (local)

Le dossier d'un run lancé par `nll_run.py` (dépôt `NonLinLoc_ToolBox`) :
`<nom>.hyp` (toutes les localisations, `PUBLIC_ID` = Event ID SeisComP),
`evenements.csv` (origine de référence de chaque événement, celle dont
viennent les pointés) et `run.yaml` (profil, période, `LOCGRID`,
`LOCSEARCH`). Les dossiers listés dans `paths.nll_runs` sont lus avec le
bulletin : chaque run local apparaît comme un run NLL de plus, sous son nom,
dans tous les scripts `nll_*` et `summary_table.py`.

Les incertitudes sont calculées comme dans SeisComP (latitude ← √CovYY,
longitude ← √CovXX, profondeur ← √CovZZ de NLL, vérifié à 1 % près sur
`1D_ubaye_v2`). Un run local a en plus le nombre de phases, le gap et la
distance minimale (colonnes `nphs`, `gap`, `dmin_km`) : les critères
`nphs_min`, `gap_max` et `dmin_max_km` des classes de qualité s'appliquent.
Les localisations rejetées par NLLoc (`REJECTED`) ne sont pas gardées.

### Run scrtdd

Le dossier qui contient `reloc-event.csv`. Les fichiers d'entrée et par
cluster (`input-*.csv`, `cluster-N-*.csv`, `xcorr.csv`) peuvent être dans ce
dossier ou dans un sous-dossier `multievent_<date>_<pid>/` : ils sont
trouvés automatiquement. Lancé avec `--reloc-catalog` (fichiers de
`scrtdd_inputs.py` de NonLinLoc_ToolBox), scrtdd n'écrit pas d'`input-*.csv` :
les positions de départ et les stations sont alors lues dans `event.csv` et
`station.csv`. L'eventlist (`eventlist*.csv`) est cherchée dans le
même dossier, sauf si `paths.scrtdd_eventlist` est renseigné.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # seulement la première fois sauf si changement dans le requirements.txt
```

## Configuration

Tous les chemins et paramètres sont centralisés dans un fichier YAML. Aucun
chemin n'est codé en dur dans les scripts.

```bash
cp config.example.yaml config.yaml
```

Puis éditer `config.yaml` :
- `paths.bulletin` : le bulletin purifié (ou une liste de bulletins)
- `paths.nll_runs` : les runs NLL locaux à analyser avec le bulletin
- `paths.scrtdd_run_dir` : le dossier du run scrtdd à analyser
- `paths.output_dir` : où écrire les résultats (défaut `outputs/`)
- `nll.author` : le run NLL à analyser (valeur de la colonne `Author`)
- `nll.quality_classes` : les seuils des classes de qualité
- `compare.*` : les runs NLL et scrtdd à comparer
- `map.zone`, `map.sections` : emprise de la carte et profils de coupe

`config.yaml` est spécifique à chaque machine et n'est **pas** versionné
(voir `.gitignore`) ; `config.example.yaml` documente chaque champ.

## Utilisation

```bash
python nll_quality.py
python nll_vs_reference.py
python nll_compare_runs.py
python dd_shifts.py
python dd_residuals.py
python dd_links.py
python dd_compare_runs.py
python summary_table.py
```

Un fichier de configuration alternatif peut être passé explicitement :

```bash
python dd_shifts.py --config config_run_sans_cc.yaml
```

## Tests

```bash
pytest tests/
```

Les tests couvrent la lecture du bulletin, le choix des origines NLL et de
référence, le calcul des écarts, le classement en qualité, les conversions
géographiques, l'association par temps, la lecture des runs NLL locaux
(`.hyp`, `run.yaml`) et la lecture des deux organisations de dossier scrtdd. `tests/test_gridding.py` vérifie le calcul du niveau
d'octree de `nll_gridding.py`.

## Structure des fichiers

```
LocQuality_ToolBox/
├── README.md               ← ce fichier
├── requirements.txt        ← dépendances Python
├── config.example.yaml     ← modèle de configuration (versionné)
├── config.yaml             ← configuration locale (non versionné)
├── loc_common.py           ← code partagé (lecture, géographie, figures)
├── nll_quality.py          ← classes de qualité NLL
├── nll_vs_reference.py     ← NLL vs référence LOCSAT
├── nll_compare_runs.py     ← comparaison de runs NLL
├── nll_gridding.py         ← quadrillage (octree) et paliers de profondeur
├── nll_station_residuals.py ← résidus par station
├── nll_sections.py         ← netteté des structures (cartes et coupes)
├── nll_coupe_interactive.py ← carte et coupe interactives
├── dd_shifts.py            ← déplacements scrtdd
├── dd_residuals.py         ← résidus scrtdd
├── dd_links.py             ← liens DD et corrélation croisée
├── dd_compare_runs.py      ← comparaison de runs scrtdd
├── summary_table.py        ← tableau récapitulatif par événement
├── tests/                  ← tests unitaires (pytest)
└── outputs/                ← résultats (non versionné)
```

## Pistes d'évolution

- Ajouter à l'export scquery le gap azimutal, le nombre de phases utilisées
  et la distance minimale (table `Origin` : `quality_azimuthalGap`,
  `quality_usedPhaseCount`, `quality_minimumDistance`) pour des classes de
  qualité plus fiables.
- Incertitudes de relocalisation réalistes par bootstrap ou jackknife :
  cela demande de relancer scrtdd sur des jeux de données rééchantillonnés.
