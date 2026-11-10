"""
Paired t-tests over seeds: HQGC versus every other model in a train_fast.py results directory (Table R3 of the
response letter). Runs with the same seed share the validation split and the initialization, so they are paired.

Example:
  python code/train_fast.py --data_dir ... --variant hqgc classical novqc nomp nonorm
  python code/train_fast.py --data_dir ... --variant classical novqc --n_qubits 16   # also 32, 64
  python code/analysis/paired_tests.py results
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def main(results_dir):
    runs = defaultdict(dict)  # (model tag, vulnerability) -> {seed: result}
    for f in glob.glob(os.path.join(results_dir, '*_s*.json')):
        r = json.load(open(f))
        tag = re.sub(rf"_{r['vulnerability']}_s{r['seed']}\.json$", '', os.path.basename(f))
        runs[tag, r['vulnerability']][r['seed']] = r

    for vuln in sorted({v for _, v in runs}):
        ref = runs.get(('hqgc', vuln))
        if not ref:
            continue
        seeds = sorted(ref)
        h = {m: np.array([ref[s][m] for s in seeds]) * 100 for m in ('accuracy', 'f1')}
        print(f"\n{vuln}: HQGC {ref[seeds[0]]['n_params']} params, acc {h['accuracy'].mean():.1f}, "
              f"F1 {h['f1'].mean():.1f} (seeds {seeds})")
        for (tag, v), res in sorted(runs.items()):
            if v != vuln or tag == 'hqgc' or sorted(res) != seeds:
                continue
            o = {m: np.array([res[s][m] for s in seeds]) * 100 for m in ('accuracy', 'f1')}
            cells = []
            for m in ('accuracy', 'f1'):
                d = h[m] - o[m]
                p = stats.ttest_rel(h[m], o[m]).pvalue if d.std() > 0 else float('nan')
                cells.append(f"{m[:3]} {o[m].mean():.1f}  HQGC-other {d.mean():+.2f} (p={p:.2f})")
            print(f"  {tag:<16s} {res[seeds[0]]['n_params']:>6d} params  " + "   ".join(cells))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results')
