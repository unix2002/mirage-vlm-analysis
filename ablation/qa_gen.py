"""QA the regen_gen output while it runs (login node, no GPU — just reads JSON).
Usage:
  python3 ablation/qa_gen.py /scratch-shared/scur0259/mirage_test_plans_gen/gen
Run it repeatedly; it only looks at whatever sample_*.json have been written so far.
"""
import glob
import json
import sys
from collections import defaultdict

d = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(f"{d}/sample_*.json"))
print(f"=== QA {d} ===")
print(f"samples done: {len(files)}")
if not files:
    sys.exit()


def pct(a, b):
    return f"{100*a/b:.0f}%" if b else "-"


clean_nonempty = 0
clean_lens = []
exact = prefix = firstmove = matchable = 0
ev_total = ev_changed = ev_empty = 0
chg_by_k = defaultdict(lambda: [0, 0])   # k -> [changed, total]
empty_by_k = defaultdict(int)
flagged_clean = []
examples = []

for f in files:
    o = json.load(open(f))
    cg = o.get("clean_plan_gen") or []
    if cg:
        clean_nonempty += 1
        clean_lens.append(len(cg))
    else:
        flagged_clean.append(o.get("sample_id"))
    ctf = o.get("clean_plan_tf")
    if ctf is not None:
        matchable += 1
        if cg == ctf:
            exact += 1
        if cg == ctf[:len(cg)] or ctf == cg[:len(ctf)]:
            prefix += 1
        if cg and ctf and cg[0] == ctf[0]:
            firstmove += 1
    for key, s in o.get("subsets", {}).items():
        if not s.get("evaluated"):
            continue
        k = key.count("1")
        ev_total += 1
        chg_by_k[k][1] += 1
        plan = s.get("ablated_plan_gen") or []
        if s.get("changed"):
            ev_changed += 1
            chg_by_k[k][0] += 1
            if len(examples) < 6:
                examples.append((o["sample_id"], key, cg, plan))
        if not plan:
            ev_empty += 1
            empty_by_k[k] += 1

mean_len = sum(clean_lens) / len(clean_lens) if clean_lens else 0
print(f"clean plan: non-empty {clean_nonempty}/{len(files)} ({pct(clean_nonempty, len(files))}) "
      f"| mean len {mean_len:.1f}")
print(f"clean_gen vs clean_tf (n={matchable}): exact {pct(exact, matchable)} "
      f"| prefix {pct(prefix, matchable)} | first-move {pct(firstmove, matchable)}")
print(f"evaluated subsets: {ev_total} | changed {ev_changed} ({pct(ev_changed, ev_total)}) "
      f"| empty-plan {ev_empty} ({pct(ev_empty, ev_total)})")
print("reroute rate by k:", " ".join(f"k{k}={pct(chg_by_k[k][0], chg_by_k[k][1])}" for k in sorted(chg_by_k)))
print("empty-plan by k:  ", " ".join(f"k{k}={empty_by_k[k]}" for k in sorted(empty_by_k)))
print("example reroutes (sid, mask, clean -> ablated):")
for sid, key, cg, plan in examples:
    print(f"  {sid} {key}: {cg} -> {plan}")
if flagged_clean:
    print(f"FLAG empty clean plan ({len(flagged_clean)}):", flagged_clean[:20])
