"""
Benchmark statistics of Table R2: class balance, duplicated graphs across the split, label ambiguity,
and the exact accuracy / F1 ceilings for any classifier whose input is the released graph.

    python code/analysis/benchmark_stats.py /path/to/GNNSCVulDetector/train_data
"""
import json, sys, os
from collections import Counter, defaultdict
base = sys.argv[1]

def key(g, typed=True):
    nf = tuple(tuple(r) for r in g["node_features"])
    e = tuple(sorted(tuple(x) if typed else (x[0], x[2]) for x in g["graph"]))
    return (nf, e)

def ceiling(test, train_keys=None, typed=True):
    groups = defaultdict(lambda: [0, 0])
    for g in test:
        groups[key(g, typed)][int(g["targets"])] += 1
    acc = sum(max(v) for v in groups.values()) / len(test)
    P = sum(v[1] for v in groups.values())
    items = sorted(groups.values(), key=lambda v: -v[1] / (v[0] + v[1]))
    best, tp, fp = 0.0, 0, 0
    for n0, n1 in items:
        tp += n1; fp += n0
        best = max(best, 2 * tp / (tp + fp + P))
    return acc, best

for vt in ["reentrancy", "timestamp"]:
    tr = json.load(open(os.path.join(base, vt, "train.json")))
    te = json.load(open(os.path.join(base, vt, "valid.json")))
    ytr = [int(g["targets"]) for g in tr]; yte = [int(g["targets"]) for g in te]
    print(f"== {vt}")
    print(f" train {len(tr)} vuln {sum(ytr)} ({100*sum(ytr)/len(tr):.1f}%)  test {len(te)} vuln {sum(yte)} ({100*sum(yte)/len(te):.1f}%)")
    print(f" feat dim {len(tr[0]['node_features'][0])}  avg nodes train {sum(len(g['node_features']) for g in tr)/len(tr):.2f} test {sum(len(g['node_features']) for g in te)/len(te):.2f} all {sum(len(g['node_features']) for g in tr+te)/len(tr+te):.2f}")
    print(f" edge types {sorted(set(e[1] for g in tr+te for e in g['graph']))}  empty-edge graphs {sum(1 for g in tr+te if not g['graph'])}")
    ktr = Counter(key(g) for g in tr); kte = Counter(key(g) for g in te)
    print(f" distinct train {len(ktr)} test {len(kte)}")
    trk = set(ktr)
    print(f" test in train {sum(1 for g in te if key(g) in trk)} ({100*sum(1 for g in te if key(g) in trk)/len(te):.1f}%)")
    lab = defaultdict(set); cnt = Counter()
    for g in tr:
        lab[key(g)].add(int(g["targets"])); cnt[key(g)] += 1
    amb = [k for k, v in lab.items() if len(v) == 2]
    print(f" ambiguous distinct train {len(amb)} covering {sum(cnt[k] for k in amb)}")
    a, f = ceiling(te); print(f" ceiling typed acc {100*a:.1f} f1 {100*f:.1f}")
    a, f = ceiling(te, typed=False); print(f" ceiling untyped acc {100*a:.1f} f1 {100*f:.1f}")
    ctr = set(g["contract_name"] for g in tr); cte = set(g["contract_name"] for g in te)
    print(f" contract names train {len(ctr)} test {len(cte)} shared {len(ctr & cte)}; test samples w/ shared name {sum(1 for g in te if g['contract_name'] in ctr)}")
