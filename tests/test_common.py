import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc_common import (
    ScrtddRun,
    classify_quality,
    dates_to_num,
    horizontal_shift_km,
    load_bulletin,
    load_config,
    match_by_time,
    nll_origins,
    pair_with_reference,
    project_on_section,
    read_bulletin,
    quality_subsets,
    read_nll_hyp,
    read_nll_phases,
    read_nll_run,
    reference_origins,
)

# Extrait au format de purify_cat.py : origine manuelle préférée + origine NLL
BULLETIN = """\
Event ID,Origin ID,Origin Time,Latitude,Latitude uncertainty,Longitude,Longitude uncertainty,Depth,Depth uncertainty,Event Type,Author,Method ID,Preferred Origin ID,Quality Standard Error
ev1,Origin/1, 2023-10-08 09:52:20 ,44.50,0.2,6.80,0.3,3.0,0.0,earthquake,MickaelLanglais,LOCSAT,Origin/1,0.30
ev1,NLL.1, 2023-10-08 09:52:19 ,44.51,0.2,6.80,0.5,4.0,0.4,earthquake,run_a,NonLinLoc,Origin/1,0.03
ev2,Origin/2, 2023-10-09 10:00:00 ,44.40,1.0,6.70,1.0,5.0,1.0,earthquake,MickaelLanglais,LOCSAT,Origin/2,0.10
ev2,NLL.2, 2023-10-09 10:00:00 ,44.40,6.0,6.70,9.0,5.0,7.0,earthquake,run_a,NonLinLoc,Origin/2,0.02
ev2,NLL.3, 2023-10-09 10:00:00 ,44.40,1.0,6.70,1.0,5.0,1.0,earthquake,run_b,NonLinLoc,Origin/2,0.02
"""

CLASSES = [
    {"name": "A", "rms_max": 0.05, "erh_max_km": 1.0, "erz_max_km": 1.0},
    {"name": "B", "rms_max": 0.10, "erh_max_km": 2.0, "erz_max_km": 2.0},
    {"name": "D"},
]


@pytest.fixture
def bulletin(tmp_path):
    path = tmp_path / "purified_test.csv"
    path.write_text(BULLETIN)
    return read_bulletin(path)


def test_load_config_missing_file_gives_explicit_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        load_config(tmp_path / "config.yaml")


def test_read_bulletin_renames_and_parses(bulletin):
    assert {"event_id", "time", "rms", "erh_km", "depth_unc_km"} <= set(bulletin.columns)
    assert bulletin["time"].iloc[0] == pd.Timestamp("2023-10-08 09:52:20", tz="UTC")
    # erh = max des incertitudes lat / lon
    assert bulletin["erh_km"].iloc[1] == pytest.approx(0.5)


def test_nll_origins_filters_on_author(bulletin):
    assert list(nll_origins(bulletin, "run_a")["origin_id"]) == ["NLL.1", "NLL.2"]
    with pytest.raises(ValueError, match="run_b"):
        nll_origins(bulletin, "inconnu")


def test_reference_is_preferred_non_nll_origin(bulletin):
    ref = reference_origins(bulletin)
    assert list(ref["origin_id"]) == ["Origin/1", "Origin/2"]


def test_pair_with_reference_shifts(bulletin):
    pairs = pair_with_reference(nll_origins(bulletin, "run_a"), reference_origins(bulletin))
    p = pairs.set_index("event_id")
    # 0.01° de latitude ≈ 1.112 km vers le nord, NLL 1 km plus profond
    assert p.loc["ev1", "dn_km"] == pytest.approx(1.112, abs=1e-3)
    assert p.loc["ev1", "de_km"] == pytest.approx(0.0, abs=1e-9)
    assert p.loc["ev1", "dz_km"] == pytest.approx(1.0)
    assert p.loc["ev2", "d3_km"] == pytest.approx(0.0, abs=1e-9)


def test_classify_quality_first_matching_class(bulletin):
    nll = nll_origins(bulletin, "run_a")
    # ev1 : erh 0.5, erz 0.4, rms 0.03 → A ; ev2 : erh 9 → dernière classe
    assert list(classify_quality(nll, CLASSES)) == ["A", "D"]


def test_classify_quality_ignores_missing_columns(bulletin):
    nll = nll_origins(bulletin, "run_a")
    classes = [{"name": "A", "gap_max": 180, "rms_max": 0.05}, {"name": "D"}]
    assert list(classify_quality(nll, classes)) == ["A", "A"]


def test_horizontal_shift_one_degree():
    de, dn = horizontal_shift_km(6.0, 44.0, 6.0, 45.0)
    assert dn == pytest.approx(111.195, abs=0.01)
    assert de == pytest.approx(0.0, abs=1e-9)
    # 1° de longitude ≈ 111.195 × cos(lat)
    de, _ = horizontal_shift_km(6.0, 44.5, 7.0, 44.5)
    assert de == pytest.approx(111.195 * np.cos(np.radians(44.5)), abs=0.01)


def test_project_on_section_east_west():
    sec = {"lon1": 6.0, "lat1": 44.5, "lon2": 7.0, "lat2": 44.5, "half_width_km": 2.0}
    along, mask = project_on_section([6.5, 6.5, 7.5], [44.5, 44.6, 44.5], sec)
    assert along[0] == pytest.approx(111.195 * np.cos(np.radians(44.5)) / 2, abs=0.05)
    # 2e point à ~11 km du profil, 3e au-delà de l'extrémité
    assert list(mask) == [True, False, False]


def test_dates_to_num_iso_and_timezone():
    # Chaînes ISO et dates avec fuseau donnent la même valeur (UTC) ; un jour = 1
    t = dates_to_num(["2026-10-16T18:19:01Z", "2026-10-17T18:19:01.5Z"])
    tz = dates_to_num(pd.to_datetime(["2026-10-16T18:19:01Z"], utc=True))
    assert t[0] == pytest.approx(tz[0])
    assert t[1] - t[0] == pytest.approx(1 + 0.5 / 86400)


def test_match_by_time_with_tolerance():
    left = pd.to_datetime(["2024-01-01 00:00:00.4", "2024-01-01 00:10:00.0"], utc=True)
    right = pd.to_datetime(["2024-01-01 00:05:00", "2024-01-01 00:00:00"], utc=True)
    assert list(match_by_time(left, right, 1.0)) == [1, -1]


def _write_scrtdd_run(root, nested):
    """Run scrtdd minimal ; nested=True → fichiers d'entrée dans multievent_*/."""
    work = root / "multievent_20260101000000_1" if nested else root
    work.mkdir(parents=True, exist_ok=True)
    (work / "input-event.csv").write_text(
        "id,isotime,latitude,longitude,depth,magnitude\n"
        "1,2024-12-01T00:45:59.15Z,44.50,6.84,4.0,0.0\n"
        "2,2024-12-01T00:47:00.44Z,44.51,6.83,4.2,0.1\n")
    (root / "reloc-event.csv").write_text(
        "id,isotime,latitude,longitude,depth,magnitude,relocated,startRms,finalRms\n"
        "1,2024-12-01T00:45:59.16Z,44.51,6.84,3.0,0.0,true,0.2,0.1\n")
    (work / "cluster-1-event.csv").write_text("id,isotime\n1,x\n2,x\n")
    # Doit être ignoré par cluster_files("event")
    (work / "cluster-1-reloc-event.csv").write_text("id,isotime\n1,x\n")
    (work / "cluster-10-event.csv").write_text("id,isotime\n3,x\n")
    (root / "eventlist_test.csv").write_text(
        "origin,event,time\nNLL.a, evA ,t\nNLL.b,evB,t\n")


@pytest.mark.parametrize("nested", [False, True])
def test_scrtdd_run_layouts(tmp_path, nested):
    _write_scrtdd_run(tmp_path, nested)
    run = ScrtddRun(tmp_path)
    assert len(run.input_events()) == 2
    names = [p.name for p in run.cluster_files("event")]
    assert names == ["cluster-1-event.csv", "cluster-10-event.csv"]
    assert run.event_clusters().to_dict() == {1: 1, 2: 1, 3: 10}


def test_scrtdd_shifts_and_ids(tmp_path):
    _write_scrtdd_run(tmp_path, nested=True)
    run = ScrtddRun(tmp_path)
    sh = run.shifts()
    assert len(sh) == 1
    assert sh["dz_km"].iloc[0] == pytest.approx(-1.0)
    assert sh["dn_km"].iloc[0] == pytest.approx(1.112, abs=1e-3)
    ids = run.seiscomp_ids()
    assert ids.set_index("id")["event_id"].to_dict() == {1: "evA", 2: "evB"}


def test_scrtdd_run_reloc_catalog_inputs(tmp_path):
    # Lancé avec --reloc-catalog : event.csv / station.csv au lieu d'input-*
    _write_scrtdd_run(tmp_path, nested=False)
    (tmp_path / "input-event.csv").rename(tmp_path / "event.csv")
    (tmp_path / "station.csv").write_text(
        "latitude,longitude,elevation,networkCode,stationCode,locationCode\n"
        "44.5,6.8,1000.0,FR,OGXX,00\n")
    run = ScrtddRun(tmp_path)
    assert len(run.input_events()) == 2
    assert run.shifts()["dz_km"].iloc[0] == pytest.approx(-1.0)
    assert list(run.stations()["stationCode"]) == ["OGXX"]


def test_scrtdd_run_requires_reloc_event(tmp_path):
    with pytest.raises(FileNotFoundError, match="reloc-event.csv"):
        ScrtddRun(tmp_path)


# Bloc .hyp NLLoc (extrait réel, raccourci) et récapitulatif d'événements de nll_run.py
HYP = """\
NLLOC "loc/part_0/loc.20241101.235555.grid0" "LOCATED" "Location completed."
PUBLIC_ID ev1
SEARCH OCTREE nInitial 8100 nEvaluated 50012 smallestNodeSide 0.104167/0.104167/0.100694 oct_tree_integral 1.2e+01 scatter_volume 1.2e+01
HYPOCENTER  x 3.28125 y 4.11458 z -2.02951  OT 56.0973  ix -1 iy -1 iz -1 MAXIMUM_LIKELIHOOD
GEOGRAPHIC  OT 2024 11 01  23 55 56.097332  Lat 44.546954 Long 6.791360 Depth 2.029514
QUALITY  Pmax 1.50474e+09 MFmin 8.69811 MFmax 10.1469 RMS 0.0395916 Nphs 5 Gap 256.252 Dist 7.52444 Mamp -9.90 0 Mdur -9.90 0
STATISTICS  ExpectX 0.729791 Y 0.258577 Z 0.762342  CovXX 0.25 XY 8.2414 XZ -11.0046 YY 1.0 YZ -1.48201 ZZ 4.0 EllAz1  104.166 Dip1  65.349 Len1  6.59096
END_NLLOC
NLLOC "loc/part_0/loc.20241101.223951.grid0" "REJECTED" "Too few phases."
PUBLIC_ID ev2
GEOGRAPHIC  OT 2024 11 01  22 39 51.900000  Lat 44.5 Long 6.8 Depth 5.0
END_NLLOC
"""

EVENTS = """\
event_id,event_type,origin_id,origin_time,method,latitude,longitude,depth,lat_unc_km,lon_unc_km,depth_unc_km,rms,n_P,n_S,gap,used_phases,source
ev1,earthquake,Origin/1,2024-11-01T23:55:55.362581Z,LOCSAT,44.49,6.88,1.57,0.5,0.4,0.0,0.12,3,2,,,2024-11-01.xml
ev2,earthquake,Origin/2,2024-11-01T22:39:51.919222Z,LOCSAT,44.50,6.83,5.53,0.6,0.5,0.0,0.10,2,0,,,2024-11-01.xml
"""


def test_read_nll_hyp(tmp_path):
    path = tmp_path / "run.hyp"
    path.write_text(HYP)
    hyp = read_nll_hyp(path)
    assert list(hyp["event_id"]) == ["ev1", "ev2"]
    assert list(hyp["status"]) == ["LOCATED", "REJECTED"]
    row = hyp.iloc[0]
    assert row["latitude"] == pytest.approx(44.546954) and row["depth"] == pytest.approx(2.029514)
    assert row["time"] == pd.Timestamp("2024-11-01 23:55:56.097332", tz="UTC")
    assert (row["nphs"], row["gap"], row["dmin_km"]) == (5, pytest.approx(256.252), pytest.approx(7.52444))
    # Incertitudes comme SeisComP : latitude ← CovYY, longitude ← CovXX
    assert (row["lat_unc_km"], row["lon_unc_km"], row["depth_unc_km"]) == (1.0, 0.5, 2.0)
    assert row["smallest_node_km"] == pytest.approx(0.104167)


def test_read_nll_run_and_load_bulletin(tmp_path, bulletin):
    run = tmp_path / "runs" / "mon_run"
    run.mkdir(parents=True)
    (run / "mon_run.hyp").write_text(HYP)
    (run / "evenements.csv").write_text(EVENTS)
    (run / "run.yaml").write_text("nom: mon_run\nlocsearch: OCT 30 30 9 0.001 50000 5000 0 1\n"
                                  "locgrid: 101 101 30 -50 -50 -5.0 1.0 1.0 1.0 PROB_DENSITY SAVE\n")
    df = read_nll_run(run)
    nll = nll_origins(df, "mon_run")
    assert list(nll["event_id"]) == ["ev1"]  # l'événement rejeté n'est pas gardé
    assert nll.iloc[0]["erh_km"] == 1.0
    ref = reference_origins(df)
    assert set(ref["origin_id"]) == {"Origin/1", "Origin/2"}
    pairs = pair_with_reference(nll, ref)
    assert pairs.iloc[0]["dz_km"] == pytest.approx(2.029514 - 1.57)

    bulletin_path = tmp_path / "purified_test.csv"
    config = {"paths": {"bulletin": str(bulletin_path), "nll_runs": [str(run)]}}
    both = load_bulletin(config)
    assert {"run_a", "run_b", "mon_run"} <= set(both["author"])
    with pytest.raises(ValueError):
        load_bulletin({"paths": {}})


def test_read_nll_phases(tmp_path):
    run = tmp_path / "mon_run"
    run.mkdir()
    (run / "run.yaml").write_text("nom: mon_run\n")
    (run / "phases.csv").write_text(
        "event_id,station,phase,time,error_s,tt_pred_s,residual_s,weight,dist_km,azimuth,ray_azimuth,ray_dip,tcorr_s\n"
        "ev1,SURF,Pg,2024-11-01T23:55:57,0.02,1.69,-0.01,1.6,5.7,173.0,173.0,139.2,0.0\n"
        "ev1,JAUF,Sg,2024-11-01T23:55:59,0.02,4.39,-0.18,0.8,12.3,200.5,200.5,112.5,0.0\n")
    ph = read_nll_phases(run)
    assert list(ph["phase_type"]) == ["P", "S"]
    assert set(ph["author"]) == {"mon_run"}


def test_quality_subsets_same_events_for_all_runs(bulletin):
    runs = {a: nll_origins(bulletin, a) for a in ["run_a", "run_b"]}
    subsets = quality_subsets(runs, CLASSES, ["A", "B"])
    # ev2 est commun aux deux runs ; il est en classe D pour run_a (erh 9 km)
    assert subsets["tous"] == {"ev2"}
    assert subsets["A_B"] == set()
