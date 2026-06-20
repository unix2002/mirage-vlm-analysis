import json
from pathlib import Path

RESULTS_PATH = Path('data/ablation_results.json')
ABLATED_PLANS_DIST_PATH = Path('data/train_plans_gen.jsonl')

_results_cache = None
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
    """Normalise sample id to integer-string key (e.g. 'sample_007' → '7')."""
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


def load_moves(sample_id, token_rank):
    entry, err = load_per_token_summary(sample_id, token_rank)
    if err or not entry:
        return None
    return entry.get('moves')


def load_ablated_plans(sample_id):
    """Return parsed ablated_plans for sample_id from the gen dist file, or None."""
    sid = int(_fmt_sample_id(sample_id))
    cache = _load_ablated_plans_dist()
    if cache and sid in cache:
        return cache[sid]
    return None


def load_clean_plan(sample_id):
    """Return the clean plan (list of directions) for a sample, or None."""
    plans = load_ablated_plans(sample_id)
    if plans is None:
        return None
    return (plans.get('clean_plan_gen')
            or plans.get('clean_plan_tf')
            or plans.get('clean_plan'))


def _load_ablated_plans_dist():
    """Lazy-load train_plans_gen.jsonl into a dict {sample_id: entry}."""
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

def mask_for_ranks(ablated_ranks, n=6):
    """Bitmask over token ranks 0..n-1 matching ablated_plans subset keys."""
    ranks = set(ablated_ranks)
    return ''.join('1' if i in ranks else '0' for i in range(n))


DEFAULT_TOKEN_LABELS = ['latent_start', 'pad_1', 'pad_2', 'pad_3', 'pad_4', 'latent_end']


def _token_label(labels, i):
    if labels and str(i) in labels:
        return labels[str(i)]
    return DEFAULT_TOKEN_LABELS[i] if i < len(DEFAULT_TOKEN_LABELS) else f'T{i}'


def token_marginal_contributions(sample_id):
    """Per-token individual (k=1) and mean marginal KL contribution from combinatorial data.

    Individual = KL when zeroing only that token.
    Marginal = average increase in KL from adding this token to subsets without it.
    Returns a list of dicts or None.
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


def sample_dose_response(sample_id):
    """Per-k KL distribution and flip rates for one sample. Returns {'rows': [...]} or None."""
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

    rows = []
    for k in sorted(by_k):
        kl = sorted(by_k[k]['kl'])
        if not kl:
            continue
        rows.append({
            'k': k,
            'kl_median': kl[len(kl) // 2],
            'kl_lo': kl[0],
            'kl_hi': kl[-1],
            'plan_flip_pct': 100.0 * by_k[k]['plan_flips'] / len(kl),
        })
    return {'rows': rows, 'token_labels': plans.get('token_labels'), 'n': n}


def current_combo_metrics(sample_id, ablated_ranks):
    """KL and plan-flip flag for the exact selected combo, or None."""
    if not ablated_ranks:
        return None
    plans = load_ablated_plans(sample_id)
    if not plans:
        return None
    n = plans.get('n', 6)
    s = plans['subsets'].get(mask_for_ranks(ablated_ranks, n))
    if s is None:
        return None
    return {
        'k': len(ablated_ranks),
        'kl': s.get('kl_mean', 0.0),
        'plan_flipped': bool(s.get('changed')),
    }


def subset_lattice(sample_id):
    """All 63 ablation subsets for one sample, for the KL-fingerprint view. Returns {'cells': [...], 'max_kl', 'n'}."""
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
            'changed': bool(s.get('changed')),
        })
    return {'cells': cells, 'max_kl': max_kl, 'n': n}
