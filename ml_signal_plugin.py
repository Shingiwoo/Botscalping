"""
ml_signal_plugin.py — Plugin ML untuk live trading (selaras backtester)
PATCH 2025-08-11: _build_latest_features kini mengembalikan DataFrame (bukan ndarray)
untuk menghindari warning: "X does not have valid feature names" saat predict.
"""
from __future__ import annotations
import time, os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from engine_core import SAFE_EPS

@dataclass
class MLParams:
    use_ml: bool = False
    score_threshold: float = 1.0
    min_train_bars: int = 400
    lookahead: int = 5
    retrain_every_bars: int = 120
    up_prob_thres: float = 0.55
    down_prob_thres: float = 0.45

class MLSignal:
    def __init__(
        self,
        coin_cfg: Dict[str, Any] | None = None,
        thr: float = 1.20,
        htf: str | None = None,
        heikin: bool = False,
        model_path: str | None = None,
        device: str = "cpu",
    ):
        cfg = coin_cfg or {}
        env = os.environ

        self.coin_cfg = cfg
        self.thr = float(thr)
        self.htf = htf
        self.heikin = bool(heikin)
        self.model_path = model_path
        self.device = device

        def _getf(key: str, default: float) -> float:
            v = cfg.get(key, env.get(key))
            try: return float(v)
            except Exception: return float(default)
        def _geti(key: str, default: int) -> int:
            v = cfg.get(key, env.get(key))
            try: return int(v)
            except Exception: return int(default)
        def _getb(key: str, default: bool) -> bool:
            v = cfg.get(key, env.get(key))
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)): return bool(int(v))
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"1","true","y","yes","on"}: return True
                if s in {"0","false","n","no","off"}: return False
            return bool(default)

        self.params = MLParams(
            use_ml=_getb('USE_ML', False),
            score_threshold=_getf('SCORE_THRESHOLD', 1.0),
            min_train_bars=_geti('ML_MIN_TRAIN_BARS', 400),
            lookahead=_geti('ML_LOOKAHEAD', 5),
            retrain_every_bars=_geti('ML_RETRAIN_EVERY', 120),
            up_prob_thres=_getf('ML_UP_PROB', 0.55),
            down_prob_thres=_getf('ML_DOWN_PROB', 0.45),
        )
        self.model: Optional[RandomForestClassifier] = None
        self._last_fit_index: int = -1
        self._last_fit_time: float = 0.0

    @property
    def use_ml(self) -> bool:
        return self.params.use_ml

    def fit_if_needed(self, df: pd.DataFrame) -> None:
        feats, target = self._build_dataset(df)
        if feats is None or target is None: return
        n = len(feats)
        if n < self.params.min_train_bars: return
        need = (self._last_fit_index < 0) or (n - self._last_fit_index >= self.params.retrain_every_bars) or (time.time() - self._last_fit_time > 60)
        if not need: return
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        m.fit(feats, target)
        self.model = m
        self._last_fit_index = n
        self._last_fit_time = time.time()

    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]:
        if self.model is None: return None
        f = self._build_latest_features(df)
        if f is None: return None
        proba = self.model.predict_proba(f)[0, 1]
        proba = min(max(float(proba), SAFE_EPS), 1 - SAFE_EPS)
        return proba

    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None) -> Tuple[bool, bool]:
        th = float(self.params.score_threshold)
        sc_long = 1.0 if base_long else 0.0
        sc_short = 1.0 if base_short else 0.0
        if self.use_ml and (up_prob is not None):
            if up_prob >= self.params.up_prob_thres:
                sc_long += 1.0
            elif up_prob <= self.params.down_prob_thres:
                sc_short += 1.0
        return (sc_long >= th, sc_short >= th)

    # ===== dataset & fitur =====
    def _build_dataset(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        if df is None or df.empty: return None, None
        d = df.copy()
        req = {'close','rsi','macd','atr'}
        if not req.issubset(d.columns): return None, None
        d['bb_width'] = (d['close'].rolling(20).std() * 4.0 / d['close']).replace([np.inf, -np.inf], 0.0)
        d['lag_ret'] = d['close'].pct_change().shift(1)
        d['vol'] = d['close'].rolling(20).std().shift(1)
        la = int(self.params.lookahead)
        d['target'] = (d['close'].shift(-la) > d['close']).astype(int)
        feat_cols = ['rsi','macd','atr','bb_width','lag_ret','vol']
        ds = d[feat_cols + ['target']].dropna()
        if ds.empty: return None, None
        X = ds[feat_cols]
        y = ds['target']
        return X, y

    def _build_latest_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty: return None
        d = df.copy()
        if not {'close','rsi','macd','atr'}.issubset(d.columns): return None
        d['bb_width'] = (d['close'].rolling(20).std() * 4.0 / d['close']).replace([np.inf, -np.inf], 0.0)
        d['lag_ret'] = d['close'].pct_change().shift(1)
        d['vol'] = d['close'].rolling(20).std().shift(1)
        feat_cols = ['rsi','macd','atr','bb_width','lag_ret','vol']
        row = d[feat_cols].iloc[[-1]].dropna()
        if row.empty: return None
        return row  # DataFrame dengan nama kolom, cocok saat fit
