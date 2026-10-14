import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loc_common import (
    ScrtddRun,
    classify_quality,
    horizontal_shift_km,
    load_config,
    match_by_time,
    nll_origins,
    pair_with_reference,
    project_on_section,
    read_bulletin,
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


def test_scrtdd_run_requires_reloc_event(tmp_path):
    with pytest.raises(FileNotFoundError, match="reloc-event.csv"):
        ScrtddRun(tmp_path)
