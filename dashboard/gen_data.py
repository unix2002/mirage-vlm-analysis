"""Free-run reroute data: per sample, per ablation subset, shows whether the
model's plan changed vs the clean free-run plan.

File: data/train_plans_gen.jsonl"""
import json
import logging
from pathlib import Path

from .callbacks.ablation_v2 import mask_for_ranks

TRAIN_GEN_PATH = Path('data/train_plans_gen.jsonl')

_train = None


def _load(path):
    cache = {}
    if path.exists():
        try:
            with open(path) as f:
                for line in f:
                    e = json.loads(line)
                    cache[int(e['sample_id'])] = e
        except Exception as ex:
            logging.warning(f"failed to load {path}: {ex}")
    return cache


def _entry(sample_id):
    global _train
    if _train is None:
        _train = _load(TRAIN_GEN_PATH)
    try:
        return _train.get(int(str(sample_id).split('_')[-1]))
    except (ValueError, IndexError):
        return None


_flippers = None


def flipper_ids():
    """Sample ids ('sample_NNN') that have at least one plan-flipping ablation subset. Cached."""
    global _flippers
    if _flippers is not None:
        return _flippers
    out = set()
    for sid, e in _load(TRAIN_GEN_PATH).items():
        if any(s.get('changed') for s in e.get('subsets', {}).values()):
            out.add(f'sample_{sid:03d}')
    _flippers = out
    return out


def reroute_plan(sample_id, ablated_ranks):
    """If this ablation changes the free-run plan, return (clean_plan_gen, new_plan);
    otherwise None (no change, no data, or the subset was pruned)."""
    if not ablated_ranks:
        return None
    e = _entry(sample_id)
    if not e:
        return None
    n = e.get('n', 6)
    s = e.get('subsets', {}).get(mask_for_ranks(ablated_ranks, n))
    if s and s.get('evaluated') and s.get('changed'):
        return e.get('clean_plan_gen') or [], s.get('ablated_plan_gen') or []
    return None
