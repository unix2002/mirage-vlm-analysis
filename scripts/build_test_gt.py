#!/usr/bin/env python3
"""Derive ground-truth VSP plans from ``map_desc`` and inject them as
``text_output`` so the test split can be run through the combo pipeline with
full parity to train.

The VSP test split (``test_direct.jsonl``) ships without ``text_output``.  Each
maze is fully specified by ``map_desc`` (0 = free, -1 = wall/hole, 1 = start,
2 = goal), so the optimal plan can be recovered with a shortest-path search.

Convention (reverse-engineered and verified against all 1000 train GT plans):
    * start  = cell containing 1
    * goal   = cell containing 2
    * UP    = (-1, 0)   DOWN = (+1, 0)
    * LEFT  = ( 0,-1)   RIGHT= ( 0,+1)
    * BFS tie-break neighbour order: UP, LEFT, RIGHT, DOWN
      -> reproduces 976/1000 train answers exactly; 1000/1000 are valid+optimal.

Usage
-----
    # 1) sanity-check the solver against train (must report ~0.976 exact, 1.0 valid)
    python build_test_gt.py --selfcheck data/vsp_spatial_planning/train_direct.jsonl

    # 2) generate text_output for the test split
    python build_test_gt.py data/vsp_spatial_planning/test_direct.jsonl \
        --out data/vsp_spatial_planning/test_direct_gt.jsonl
"""
import argparse
import json
import re
from collections import deque

DELTAS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
TIE_BREAK = ('UP', 'LEFT', 'RIGHT', 'DOWN')
START_VAL, GOAL_VAL, WALL_VAL = 1, 2, -1

# train_direct_with_oi.jsonl prepends this to text_output so Mirage's
# place_output_image() can expand <output_image> into latent image-pad tokens.
# (place_output_image is pure text; the actual image_output file is not needed.)
WITH_OI_PREFIX = '<think></think><output_image>'


def _find(grid, val):
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == val:
                return (i, j)
    return None


def solve(grid):
    """Return the optimal plan as a list of direction words, or None."""
    start, goal = _find(grid, START_VAL), _find(grid, GOAL_VAL)
    if start is None or goal is None:
        return None
    R, C = len(grid), len(grid[0])
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for d in TIE_BREAK:
            dr, dc = DELTAS[d]
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] != WALL_VAL and (nr, nc) not in prev:
                prev[(nr, nc)] = (cur, d)
                q.append((nr, nc))
    if goal not in prev:
        return None
    path, node = [], goal
    while prev[node] is not None:
        parent, d = prev[node]
        path.append(d)
        node = parent
    return path[::-1]


def to_boxed(plan):
    return r'\boxed{' + ', '.join(plan) + '}'


def _gt_moves(text_output):
    m = re.search(r'\\boxed\{(.+?)\}', text_output or '')
    return [t.strip() for t in re.split(r'[ ,]+', m.group(1)) if t.strip()] if m else []


def _valid(grid, plan):
    cur = _find(grid, START_VAL)
    R, C = len(grid), len(grid[0])
    for mv in plan:
        dr, dc = DELTAS[mv]
        cur = (cur[0] + dr, cur[1] + dc)
        if not (0 <= cur[0] < R and 0 <= cur[1] < C) or grid[cur[0]][cur[1]] == WALL_VAL:
            return False
    return cur == _find(grid, GOAL_VAL)


def selfcheck(path):
    rows = [json.loads(l) for l in open(path)]
    exact = valid = lenok = n = 0
    for r in rows:
        gt = _gt_moves(r.get('text_output', ''))
        if not gt:
            continue
        n += 1
        plan = solve(r['map_desc'])
        if plan is None:
            continue
        exact += (plan == gt)
        valid += _valid(r['map_desc'], gt)
        lenok += (len(plan) == len(gt))
    print(f"self-check on {n} train mazes with ground truth:")
    print(f"  exact reproduction : {exact}/{n}  ({exact/n:.3f})")
    print(f"  GT valid+optimal   : {valid}/{n} valid, {lenok}/{n} optimal-length")
    print("  expected ~0.976 exact, 1.000 valid, 1.000 optimal")


def generate(in_path, out_path, with_oi=False):
    rows = [json.loads(l) for l in open(in_path)]
    solved = unsolved = 0
    with open(out_path, 'w') as f:
        for r in rows:
            plan = solve(r['map_desc'])
            if plan is None:
                unsolved += 1
                r['text_output'] = None
            else:
                solved += 1
                boxed = to_boxed(plan)
                r['text_output'] = (WITH_OI_PREFIX + boxed) if with_oi else boxed
            f.write(json.dumps(r) + '\n')
    kind = 'with_oi' if with_oi else 'plain'
    print(f"wrote {out_path} ({kind}): {solved} solved, {unsolved} unsolved (of {len(rows)})")
    if unsolved:
        print("  WARNING: unsolved mazes have text_output=None — inspect before the combo run")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='jsonl with map_desc (test_direct.jsonl)')
    ap.add_argument('--out', help='output jsonl with text_output injected')
    ap.add_argument('--selfcheck', action='store_true',
                    help='validate the solver against a file that already has text_output (train)')
    ap.add_argument('--with_oi', action='store_true',
                    help="prepend '<think></think><output_image>' to text_output (combo-pipeline input format)")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(args.input)
    elif args.out:
        generate(args.input, args.out, with_oi=args.with_oi)
    else:
        ap.error('provide --out (to generate) or --selfcheck (to validate)')
