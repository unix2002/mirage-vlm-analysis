"""Free-run reroute data (regen_gen output).

Per sample, per ablation subset: the greedily generated plan and whether it
changed vs the clean free-run plan. Lets the dashboard show, when an ablation
reroutes the model's plan, the new plan in red.

Files (jsonl, one record per sample, keyed by int sample_id):
  data/test_plans_gen.jsonl   -> test_NNN samples
  data/train_plans_gen.jsonl  -> sample_NNN samples
"""
import json
import logging
from pathlib import Path

TEST_GEN_PATH = Path('data/test_plans_gen.jsonl')
TRAIN_GEN_PATH = Path('data/train_plans_gen.jsonl')

_test = None
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
    global _test, _train
    sid = str(sample_id)
    if sid.startswith('test_'):
        if _test is None:
            _test = _load(TEST_GEN_PATH)
        cache = _test
    else:
        if _train is None:
            _train = _load(TRAIN_GEN_PATH)
        cache = _train
    try:
        return cache.get(int(sid.split('_')[-1]))
    except (ValueError, IndexError):
        return None


def _mask(ranks, n):
    rs = set(ranks)
    return ''.join('1' if i in rs else '0' for i in range(n))


_flippers = None


def flipper_ids():
    """Namespaced sample ids ('sample_NNN' for train, 'test_NNN' for test) that have
    at least one ablation subset which reroutes the free-run plan. Cached."""
    global _flippers
    if _flippers is not None:
        return _flippers
    out = set()
    for prefix, cache in (('sample_', _load(TRAIN_GEN_PATH)), ('test_', _load(TEST_GEN_PATH))):
        for sid, e in cache.items():
            if any(s.get('changed') for s in e.get('subsets', {}).values()):
                out.add(f'{prefix}{sid:03d}')
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
    s = e.get('subsets', {}).get(_mask(ablated_ranks, n))
    if s and s.get('evaluated') and s.get('changed'):
        return e.get('clean_plan_gen') or [], s.get('ablated_plan_gen') or []
    return None
