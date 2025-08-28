import importlib.util
import importlib.machinery
import os
import sys

def test_shim_importable():
    """
    Pastikan file lama 'Adapter SDWS.py' masih bisa diimport tanpa error
    (kompatibilitas), meskipun nama modul mengandung spasi. Kita muat lewat path.
    """
    base = os.path.dirname(os.path.dirname(__file__))  # repo/tests -> repo
    path = os.path.join(base, "indicators", "supplyanddemand", "Adapter SDWS.py")
    assert os.path.exists(path), "File shim tidak ditemukan"

    spec = importlib.util.spec_from_file_location("adapter_sdws_shim", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adapter_sdws_shim"] = mod
    spec.loader.exec_module(mod)  # type: ignore

    # Harus punya simbol utama adapter
    assert hasattr(mod, "SDWSRajaDollarAdapter")
    assert hasattr(mod, "AsyncQueuePublisher")
    assert hasattr(mod, "BasePublisher")
