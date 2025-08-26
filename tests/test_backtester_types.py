import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from ta.momentum import RSIIndicator
from indicators.sr_utils import near_level, ltf_momentum_ok


def test_near_level_returns_bool():
    out = near_level(1.0, np.array([0.99, 1.01]), 2.0)
    assert isinstance(out, bool)


def test_ltf_momentum_ok_tuple():
    df = pd.DataFrame({'close': [1, 1.01, 1.02, 1.03, 1.04, 1.05]})
    out = ltf_momentum_ok(df)
    assert isinstance(out[0], bool) and isinstance(out[1], bool)


def _extract_breakeven_block():
    source = Path('backtester_scalping.py').read_text()
    tree = ast.parse(source)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'apply_breakeven_sl':
            cur = parent.get(node)
            while cur and not (isinstance(cur, ast.If) and 'position_type is not None' in ast.unparse(cur.test)):
                cur = parent.get(cur)
            if cur:
                first = cur.body[0] if cur.body else None
                if isinstance(first, ast.If):
                    new_if = ast.If(test=cur.test, body=[first], orelse=[])
                    mod = ast.Module([new_if], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    return compile(mod, filename='tmp', mode='exec')
    raise RuntimeError('breakeven block not found')


def test_breakeven_guard():
    code = _extract_breakeven_block()

    def run(pos):
        called = {'flag': False}

        def fake_apply_breakeven_sl(**kwargs):
            called['flag'] = True
            return kwargs['sl']

        env = {
            'apply_breakeven_sl': fake_apply_breakeven_sl,
            'in_position': True,
            'position_type': pos,
            'entry': 1.0,
            'qty': 1.0,
            'sl': 0.9,
            'price': 1.0,
            'use_breakeven': True,
            'sym_cfg': {},
            'be_trigger_pct': 0.0,
        }
        exec(code, env)
        return called['flag']

    assert run(None) is False
    assert run('LONG') is True
