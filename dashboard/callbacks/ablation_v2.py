import json
from pathlib import Path

ABLATION_V2_DIR = Path('data/ablation_v2')
RESULTS_PATH = ABLATION_V2_DIR / 'results.json'
COMBOS_ALL6_PATH = ABLATION_V2_DIR / 'combos_all6.json'

_results_cache = None
_combos_all6_cache = None


def _load_results():
    global _results_cache
    if _results_cache is not None:
        return _results_cache
    if not RESULTS_PATH.exists():
        _results_cache = {}
        return _results_cache
    with open(RESULTS_PATH) as f:
        _results_cache = json.load(f)
    return _results_cache


def _fmt_sample_id(sample_id):
    if isinstance(sample_id, int):
        return str(sample_id)
    if isinstance(sample_id, str) and sample_id.startswith("sample_"):
        return sample_id[7:]
    return str(sample_id)


def load_per_token_summary(sample_id, token_rank):
    results = _load_results()
    if not results:
        return None, f"results.json not found at {RESULTS_PATH}"
    per_sample = results.get('per_sample', {})
    mode = f'zero_token_{token_rank}'
    mode_data = per_sample.get(mode)
    if mode_data is None:
        return None, f"mode '{mode}' not found in results"
    sid = _fmt_sample_id(sample_id)
    entry = mode_data.get(sid)
    if entry is None:
        return None, f"sample {sid} not found in mode '{mode}'"
    return entry, None


def load_combo_file(sample_id):
    global _combos_all6_cache
    if _combos_all6_cache is None:
        if not COMBOS_ALL6_PATH.exists():
            return None
        _combos_all6_cache = json.loads(COMBOS_ALL6_PATH.read_text())
    combos = _combos_all6_cache.get('per_sample', {}).get('combos', {})
    sid = _fmt_sample_id(sample_id)
    return combos.get(sid)


_token_pos_cache = {}


def load_moves(sample_id, token_rank):
    entry, err = load_per_token_summary(sample_id, token_rank)
    if err or not entry:
        return None
    return entry.get('moves')


def _get_token_positions(sample_id):
    """Return dict {token_rank: sequence_position} for a sample."""
    sid = _fmt_sample_id(sample_id)
    if sid in _token_pos_cache:
        return _token_pos_cache[sid]

    results = _load_results()
    if not results:
        return None

    per_sample = results.get('per_sample', {})
    pos_map = {}
    for rank in range(6):
        mode = f'zero_token_{rank}'
        entry = per_sample.get(mode, {}).get(sid)
        if entry:
            tp = entry.get('token_pos')
            if tp is not None:
                pos_map[rank] = tp

    _token_pos_cache[sid] = pos_map
    return pos_map


def build_bitmask(sample_id, ablated_ranks):
    """
    ablated_ranks: list of token ranks to ablate, e.g. [1, 3]  (T1, T3)
    Returns (mask_string, combo_data) or (None, error_string)
    """
    combo = load_combo_file(sample_id)
    if combo is None:
        return None, f"Combo data not found for sample {sample_id}"
    positions = combo['positions']
    token_pos_map = _get_token_positions(sample_id)
    if token_pos_map is None:
        return None, "Per-token position data not available"

    bits = ['0'] * len(positions)
    for rank in ablated_ranks:
        pos = token_pos_map.get(rank)
        if pos is None:
            return None, f"Token T{rank} position unknown for this sample"
        try:
            idx = positions.index(pos)
        except ValueError:
            return None, f"Token T{rank} (pos {pos}) not in visible positions {positions}"
        bits[idx] = '1'

    mask = ''.join(bits)
    if mask not in combo['subsets']:
        return None, f"Mask '{mask}' not found in combo subsets"
    return mask, combo['subsets'][mask]
