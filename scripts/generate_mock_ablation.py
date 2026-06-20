import json, random, numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUT = Path('data/ablation_v2')
N_SAMPLES = 996
MOVE_WORDS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
N_TOKENS = 6


def _mock_moves(flip=True):
    n_rows = random.randint(2, 4)
    moves = []
    flipped = False
    for r in range(n_rows):
        clean = {w: round(float(v), 4) for w, v in
                 zip(MOVE_WORDS, np.random.dirichlet(np.ones(4) * 0.3))}
        if flip and not flipped and random.random() < 0.4:
            # Flip: different top prediction
            abl = {w: round(float(v), 4) for w, v in
                   zip(MOVE_WORDS, np.random.dirichlet(np.ones(4) * 0.3))}
            ct = max(clean, key=clean.get)
            at = max(abl, key=abl.get)
            if ct != at:
                flipped = True
                fl = {"row": 433 + r, "gt": ct,
                      "clean_top": [ct, round(clean[ct], 4)],
                      "ablated_top": [at, round(abl[at], 4)]}
                moves.append({"row": 433 + r, "gt": ct,
                              "clean": clean, "ablated": abl})
                continue
        # No flip: same top prediction
        top = max(clean, key=clean.get)
        abl = dict(clean)  # copy
        moves.append({"row": 433 + r, "gt": top,
                      "clean": clean, "ablated": abl})
    return moves, fl if flipped else None


for sid in range(N_SAMPLES):
    for rank in range(N_TOKENS):
        pos = 433 + rank
        # Shared: kl_mean should vary by token rank (later tokens = higher)
        base_kl = 0.05 + rank * 0.12 + random.uniform(-0.03, 0.03)
        top1 = max(0.0, min(1.0, 1.0 - rank * 0.08 + random.uniform(-0.05, 0.05)))
        acc = max(0.0, min(1.0, 1.0 - rank * 0.06 + random.uniform(-0.05, 0.05)))

        moves, flip = _mock_moves(flip=(rank >= 2))

        summary = {
            "sample_id": sid,
            "mode": f"zero_token_{rank}",
            "kl_mean": round(base_kl, 6),
            "kl_max": round(base_kl * 2.5, 6),
            "kl_sum": round(base_kl * 7, 6),
            "top1_agreement": round(top1, 4),
            "gt_nll_clean": round(2.5 + random.uniform(-0.5, 0.5), 4),
            "gt_nll_ablated": round(2.5 + random.uniform(-0.5, 0.5), 4),
            "gt_nll_delta": round(random.uniform(-0.5, 0.5), 4),
            "gt_acc_clean": round(0.78 + random.uniform(-0.1, 0.1), 4),
            "gt_acc_ablated": round(acc, 4),
            "exact_match_clean": False,
            "exact_match_ablated": False,
            "num_answer_positions": 7,
            "token_rank": rank,
            "token_pos": pos,
            "moves": moves,
        }
        if flip:
            summary["flip"] = flip

        sample_id = f"sample_{sid:03d}"
        path = OUT / f'zero_token_{rank}' / sample_id / 'summary.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

print(f"Generated mock per-token data for {N_SAMPLES} samples × {N_TOKENS} tokens → {OUT}")
