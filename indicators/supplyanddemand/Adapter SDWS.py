# Kompatibilitas: re-export dari modul baru, aman di-load via path (tanpa parent package)
import warnings as _warnings
try:
    # Jika dimuat sebagai bagian dari package normal
    from .adapter_sdws import *  # type: ignore  # noqa: F401,F403
except Exception:
    # Jika dimuat via import path langsung (tanpa parent package)
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _dir = _os.path.dirname(__file__)
    _path = _os.path.join(_dir, "adapter_sdws.py")
    _name = "indicators.supplyanddemand.adapter_sdws"
    _spec = _ilu.spec_from_file_location(_name, _path)
    if _spec and _spec.loader:  # pragma: no cover
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules[_name] = _mod
        _sys.modules["adapter_sdws"] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore
        # Ekspor simbol publik utama
        for _name in [
            "SDWSRajaDollarAdapter",
            "AsyncQueuePublisher",
            "LoggingPublisher",
            "TelegramPublisher",
            "BasePublisher",
            "AdapterConfig",
        ]:
            if hasattr(_mod, _name):
                globals()[_name] = getattr(_mod, _name)
_warnings.warn(
    "Module 'Adapter SDWS.py' adalah shim. Ganti import ke "
    "'indicators.supplyanddemand.adapter_sdws' untuk ke depannya.",
    UserWarning,
    stacklevel=2,
)
