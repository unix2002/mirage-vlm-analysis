import json
from pathlib import Path

ABLATION_V2_DIR = Path('data/ablation_v2')
RESULTS_PATH = Path('data/ablation_results.json')
COMBOS_ALL6_PATH = Path('data/ablation_v2/subsets.json')
ABLATED_PLANS_DIR = ABLATION_V2_DIR / 'ablated_plans'
ABLATED_PLANS_DIST_PATH = Path('data/ablated_plans_dist.jsonl')

_results_cache = None
_combos_all6_cache = None
_ablated_plans_dist_cache = None


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
    """Normalise a sample id to the bare-integer key used in the JSON result
    files (e.g. 'sample_007' -> '7', 7 -> '7').

    The ablated_plans loader re-pads this to the zero-padded filename, so the
    same canonical form works for both dict-key and file-path lookups.
    """
    if isinstance(sample_id, int):
        return str(sample_id)
    s = sample_id[7:] if sample_id.startswith("sample_") else str(sample_id)
    return str(int(s)) if s.isdigit() else s


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
    _ensure_combos_cached()
    if _combos_all6_cache is None:
        return None
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


def load_ablated_plans(sample_id):
    """Return parsed ablated_plans file for sample_id, or None."""
    sid = int(_fmt_sample_id(sample_id))
    cache = _load_ablated_plans_dist()
    if cache and sid in cache:
        return cache[sid]
    
    # Fallback to file-based lookup
    str_sid = _fmt_sample_id(sample_id)
    path = ABLATED_PLANS_DIR / f'sample_{str_sid.zfill(3) if str_sid.isdigit() else str_sid}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_clean_plan(sample_id):
    """Return the clean plan (list of directions) for a sample, or None."""
    plans = load_ablated_plans(sample_id)
    if plans is None:
        return None
    return plans.get('clean_plan')


def load_visual_zero_kl(sample_id):
    """Return the visual_zero KL mean for a sample from combos_all6.json."""
    sid = _fmt_sample_id(sample_id)
    if _combos_all6_cache is None:
        if not COMBOS_ALL6_PATH.exists():
            return None
        _ensure_combos_cached()
    per_sample = _combos_all6_cache.get('per_sample', {})
    vz = per_sample.get('visual_zero', {}).get(sid)
    if vz is None:
        return None
    return vz.get('kl_mean')


def load_reference_metrics(sample_id, ablated_ranks):
    """Return (our_kl, vzero_kl) for the reference meter, or (None, None)."""
    our_kl = 0.0
    if len(ablated_ranks) == 1:
        entry, _ = load_per_token_summary(sample_id, ablated_ranks[0])
        if entry:
            our_kl = entry.get('kl_mean', 0.0)
    elif len(ablated_ranks) > 1:
        _, result = build_bitmask(sample_id, ablated_ranks)
        if result:
            our_kl = result.get('kl_mean', 0.0)
    vzero_kl = load_visual_zero_kl(sample_id)
    return our_kl, vzero_kl


def _ensure_combos_cached():
    global _combos_all6_cache
    if _combos_all6_cache is None and COMBOS_ALL6_PATH.exists():
        _combos_all6_cache = json.loads(COMBOS_ALL6_PATH.read_text())


def _load_ablated_plans_dist():
    """Lazy-load ablated_plans_dist.jsonl into a dict {sample_id: entry}."""
    global _ablated_plans_dist_cache
    if _ablated_plans_dist_cache is not None:
        return _ablated_plans_dist_cache
    if not ABLATED_PLANS_DIST_PATH.exists():
        _ablated_plans_dist_cache = {}
        return _ablated_plans_dist_cache
    cache = {}
    with open(ABLATED_PLANS_DIST_PATH) as f:
        for line in f:
            entry = json.loads(line)
            cache[entry['sample_id']] = entry
    _ablated_plans_dist_cache = cache
    return cache


def load_combo_dist(sample_id, mask):
    """Return (moves_clean, moves_ablated, kl_mean, ablated_plan) for a
    sample's combo, or (None, None, None, None) if not found.
    """
    cache = _load_ablated_plans_dist()
    if not cache:
        return None, None, None, None
    sid = int(_fmt_sample_id(sample_id))
    entry = cache.get(sid)
    if entry is None:
        return None, None, None, None
    subset = entry.get('subsets', {}).get(mask)
    if subset is None:
        return None, None, None, None
    return (
        entry.get('moves_clean', []),
        subset.get('moves_ablated', []),
        subset.get('kl_mean', 0),
        subset.get('ablated_plan', []),
    )


# ── Ablation landscape: dose-response + per-token contributions ────────
# In ablated_plans subsets the mask is indexed by token rank (bit i = rank i),
# so it can be built directly from the selected ranks without a position map.

DEFAULT_TOKEN_LABELS = ['latent_start', 'pad_1', 'pad_2', 'pad_3', 'pad_4', 'latent_end']


def mask_for_ranks(ablated_ranks, n=6):
    """Bitmask over token ranks 0..n-1 matching ablated_plans subset keys."""
    ranks = set(ablated_ranks)
    return ''.join('1' if i in ranks else '0' for i in range(n))


def _token_label(labels, i):
    if labels and str(i) in labels:
        return labels[str(i)]
    return DEFAULT_TOKEN_LABELS[i] if i < len(DEFAULT_TOKEN_LABELS) else f'T{i}'


def sample_dose_response(sample_id):
    """Per-k KL distribution and flip rates for one sample.

    KL band + plan-flip rate come from the 63 self-consistent ablated_plans
    subsets; exact-match flip rate comes from combos_all6 (bucketed by popcount).
    Returns {'rows': [...], 'token_labels', 'n'} or None.
    """
    plans = load_ablated_plans(sample_id)
    if not plans:
        return None
    n = plans.get('n', 6)
    by_k = {}
    for mask, s in plans['subsets'].items():
        b = by_k.setdefault(mask.count('1'), {'kl': [], 'plan_flips': 0})
        b['kl'].append(s.get('kl_mean', 0.0))
        if s.get('changed'):
            b['plan_flips'] += 1

    em_by_k = {}
    combo = load_combo_file(sample_id)
    if combo:
        for mask, s in combo.get('subsets', {}).items():
            c = em_by_k.setdefault(mask.count('1'), [0, 0])
            c[1] += 1
            if not s.get('em_abl', True):
                c[0] += 1

    rows = []
    for k in sorted(by_k):
        kl = sorted(by_k[k]['kl'])
        if not kl:
            continue
        em = em_by_k.get(k)
        rows.append({
            'k': k,
            'kl_median': kl[len(kl) // 2],
            'kl_lo': kl[0],
            'kl_hi': kl[-1],
            'plan_flip_pct': 100.0 * by_k[k]['plan_flips'] / len(kl),
            'em_flip_pct': (100.0 * em[0] / em[1]) if em and em[1] else None,
        })
    return {'rows': rows, 'token_labels': plans.get('token_labels'), 'n': n}


def token_marginal_contributions(sample_id):
    """Per-token individual (k=1) and mean marginal KL contribution.

    Marginal = average increase in KL from adding this token to a subset that
    does not already contain it. Returns a list of dicts or None.
    """
    plans = load_ablated_plans(sample_id)
    if not plans:
        return None
    n = plans.get('n', 6)
    labels = plans.get('token_labels')
    klof = {m: s.get('kl_mean', 0.0) for m, s in plans['subsets'].items()}

    out = []
    for i in range(n):
        single = '0' * i + '1' + '0' * (n - i - 1)
        deltas = [klof[m[:i] + '1' + m[i + 1:]] - kl
                  for m, kl in klof.items()
                  if m[i] == '0' and (m[:i] + '1' + m[i + 1:]) in klof]
        out.append({
            'rank': i,
            'label': _token_label(labels, i),
            'individual': klof.get(single, 0.0),
            'marginal': sum(deltas) / len(deltas) if deltas else 0.0,
        })
    return out


def current_combo_metrics(sample_id, ablated_ranks):
    """KL and both flip flags for the exact selected combo, or None."""
    if not ablated_ranks:
        return None
    plans = load_ablated_plans(sample_id)
    if not plans:
        return None
    n = plans.get('n', 6)
    s = plans['subsets'].get(mask_for_ranks(ablated_ranks, n))
    if s is None:
        return None
    res = {
        'k': len(ablated_ranks),
        'kl': s.get('kl_mean', 0.0),
        'plan_flipped': bool(s.get('changed')),
        'em_flipped': None,
    }
    _, combo = build_bitmask(sample_id, ablated_ranks)
    if isinstance(combo, dict):
        res['em_flipped'] = not combo.get('em_abl', True)
    return res


def ablation_status(ranks):
    """Human-readable summary of the current ablated set."""
    ranks = sorted(ranks or [])
    if not ranks:
        return "no tokens ablated"
    return "ablated: " + " + ".join(f"T{r}" for r in ranks)


def toggle_rank(sample_id, current_ranks, rank):
    """Toggle a token rank in/out of the ablated set.

    Additions are validated against the combinatorial study (only tokens whose
    position is in combos_all6 can be combined). Returns (new_ranks, status).
    """
    ranks = list(current_ranks or [])
    if rank in ranks:
        ranks.remove(rank)
        return sorted(ranks), ablation_status(ranks)

    combo = load_combo_file(sample_id)
    if combo is None:
        return sorted(ranks), "no combinatorial data for this sample"
    token_pos_map = _get_token_positions(sample_id)
    pos = token_pos_map.get(rank) if token_pos_map else None
    if pos is None or pos not in combo['positions']:
        return sorted(ranks), f"T{rank} is not in the combinatorial study"

    ranks.append(rank)
    return sorted(ranks), ablation_status(ranks)


def subset_lattice(sample_id):
    """Every ablation subset for one sample, for the KL-fingerprint view.

    Returns {'cells': [{'mask', 'k', 'kl', 'ranks'}, ...], 'max_kl', 'n'} or None.
    Built straight from the 63 self-consistent ablated_plans subsets (no rerun).
    """
    plans = load_ablated_plans(sample_id)
    if not plans:
        return None
    n = plans.get('n', 6)
    cells, max_kl = [], 0.0
    for mask, s in plans['subsets'].items():
        kl = s.get('kl_mean', 0.0)
        max_kl = max(max_kl, kl)
        cells.append({
            'mask': mask,
            'k': mask.count('1'),
            'kl': kl,
            'ranks': [i for i, b in enumerate(mask) if b == '1'],
        })
    return {'cells': cells, 'max_kl': max_kl, 'n': n}
