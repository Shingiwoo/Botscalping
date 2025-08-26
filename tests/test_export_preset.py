import os, json, tempfile, shutil
from optimizer.exporter import export_params_to_preset_json


def test_export_preset_json_merge_and_overwrite():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "presets", "scalping_params.json")
        # pertama kali (file belum ada)
        data = export_params_to_preset_json(path, "ADAUSDT_15m", {"ema_len": 34}, merge=True)
        assert "ADAUSDT_15m" in data and data["ADAUSDT_15m"]["ema_len"] == 34
        # merge key lain
        data = export_params_to_preset_json(path, "DOGEUSDT_15m", {"ema_len": 22}, merge=True)
        assert "DOGEUSDT_15m" in data and data["DOGEUSDT_15m"]["ema_len"] == 22
        # overwrite: hanya satu key yang tersisa
        data = export_params_to_preset_json(path, "XRPUSDT_15m", {"ema_len": 50}, merge=False)
        assert list(data.keys()) == ["XRPUSDT_15m"]
        assert data["XRPUSDT_15m"]["ema_len"] == 50
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

