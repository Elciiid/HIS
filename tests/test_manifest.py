"""Real-data contract: a valid manifest loads; malformed inputs fail loudly."""
import copy
import json

import numpy as np
import pytest

from hydrointel.config import DomainConfig
from hydrointel.io.loaders import InputDataError
from hydrointel.io.manifest import load


def _asc(path, data, cell, xll, yll, nodata=-9999):
    rows = "\n".join(" ".join(f"{v:.3f}" for v in r) for r in data[::-1])   # file row 0 = north
    path.write_text(f"ncols {data.shape[1]}\nnrows {data.shape[0]}\nxllcorner {xll}\nyllcorner {yll}\n"
                    f"cellsize {cell}\nNODATA_value {nodata}\n{rows}\n", encoding="utf-8")


@pytest.fixture()
def dataset(tmp_path):
    cell, ny, nx = 50.0, 40, 40
    xll, yll = 400000.0, 1200000.0
    y = (np.arange(ny) + 0.5) * cell
    dem = np.tile((ny * cell - y)[:, None] / 200.0 - 1.0, (1, nx))           # falls to the north, sea below 0
    lu = np.where(dem <= 0, 1, 3).astype(float)
    lu[:, 20] = np.where(dem[:, 20] <= 0, 1, 1)                                # a water channel column
    _asc(tmp_path / "dem.asc", dem, cell, xll, yll)
    _asc(tmp_path / "lu.asc", lu, cell, xll, yll)
    with (tmp_path / "ch.csv").open("w") as f:
        f.write("reach_id,node_id,x,y,bed_elev,width,side_slope,manning\n")
        for k, yy in enumerate(np.arange(100, 1900, 100.0)):
            f.write(f"r1,n{k},{xll + 20.5 * cell},{yll + yy},{-2.5 - k * 0.05},10,1.0,0.03\n")
    (tmp_path / "rain.csv").write_text("datetime,intensity\n2026-01-01T00:00,10\n2026-01-01T00:05,20\n")
    (tmp_path / "tide.csv").write_text("datetime,elevation\n2026-01-01T00:00,0.2\n2026-01-01T01:00,0.4\n")
    (tmp_path / "idf.csv").write_text("return_period_yr,duration_min,intensity\n10,10,100\n10,60,40\n")
    man = {"dataset_id": "unit_test_v1", "crs": "EPSG:32651", "cell_size_m": cell,
           "bounds_m": [xll, yll, xll + nx * cell, yll + ny * cell], "vertical_datum": "MSL",
           "layers": {"dem": {"file": "dem.asc", "units": "m", "nodata": -9999},
                      "landuse": {"file": "lu.asc", "units": "class", "legend": {"1": "open water", "3": "residential"}},
                      "channel_net": {"file": "ch.csv", "units": "m"},
                      "rain_gauge": {"file": "rain.csv", "units": "mm/h"},
                      "tide_gauge": {"file": "tide.csv", "units": "m_MSL"},
                      "idf_curve": {"file": "idf.csv", "units": "mm/h"}}}
    return tmp_path, man, DomainConfig(lx_m=nx * cell, ly_m=ny * cell, cell_m=cell)


def _write(tmp, man):
    p = tmp / "manifest.json"
    p.write_text(json.dumps(man), encoding="utf-8")
    return p


def test_valid_manifest_loads(dataset):
    tmp, man, dcfg = dataset
    dom, real = load(_write(tmp, man), dcfg)
    assert dom.source == "unit_test_v1" and dom.shape == (40, 40)
    assert dom.sea[-1].all() and not dom.sea[0].any()           # north edge is sea
    assert dom.network.n_reaches == 1 and dom.network.reaches[0].down == ("stage",)
    assert np.all(dom.dem2d[dom.channel] >= dom.dem[dom.channel])


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m.update(crs="EPSG:4326"), "crs"),
    (lambda m: m.update(vertical_datum="EGM2008"), "vertical_datum"),
    (lambda m: m.update(cell_size_m=25.0), "cell_size_m"),
    (lambda m: m["layers"]["dem"].update(units="ft"), "units"),
    (lambda m: m["layers"].pop("tide_gauge"), "tide_gauge"),
    (lambda m: m["layers"]["landuse"]["legend"].pop("3"), "legend"),
    (lambda m: m["layers"]["dem"].update(file="missing.asc"), "does not exist"),
    (lambda m: m.update(bounds_m=[0.0, 0.0, 2000.0, 2000.0]), "cover"),
])
def test_malformed_manifest_fails_loudly(dataset, mutate, match):
    tmp, man, dcfg = dataset
    bad = copy.deepcopy(man)
    mutate(bad)
    with pytest.raises(InputDataError, match=match):
        load(_write(tmp, bad), dcfg)


def test_dem_nodata_inside_domain_is_an_error(dataset):
    tmp, man, dcfg = dataset
    txt = (tmp / "dem.asc").read_text().splitlines()
    parts = txt[10].split()
    parts[5] = "-9999"
    txt[10] = " ".join(parts)
    (tmp / "dem.asc").write_text("\n".join(txt) + "\n")
    with pytest.raises(InputDataError, match="nodata"):
        load(_write(tmp, man), dcfg)
