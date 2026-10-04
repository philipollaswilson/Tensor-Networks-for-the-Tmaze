"""Aggregate per-seed JSON results of seed_protocol.py into mean/SD tables.

Run from the repo root: python src/seed_protocol_aggregate.py [results/seed_protocol]
"""
import glob
import json
import pathlib
import sys

import numpy as np

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'results/seed_protocol')
for maze in ['minimal', 'full']:
    files = sorted(glob.glob(str(d / f'{maze}_seed*.json')))
    if not files:
        continue
    rs = [json.load(open(f)) for f in files]
    keys = [k for k, v in rs[0].items() if isinstance(v, (int, float)) and k not in ('seed', 'init_seed')]
    print(f'\n== {maze} maze: {len(rs)} seeds ({[r["seed"] for r in rs]}) ==')
    print(f'{"metric":38s} {"mean":>10s} {"sd":>8s}   per seed')
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in rs], float)
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary[k] = (float(vals.mean()), float(sd))
        print(f'{k:38s} {vals.mean():10.4f} {sd:8.4f}   ' + ' '.join(f'{v:.4f}' for v in vals))
    json.dump({'n_seeds': len(rs), 'seeds': [r['seed'] for r in rs], 'mean_sd': summary},
              open(d / f'{maze}_summary.json', 'w'), indent=1)
    # seed-0 fidelity matrix for the figure/caption
    r0 = [r for r in rs if r['seed'] == 0]
    if r0:
        print('histories:', r0[0]['histories'])
        print(np.array(r0[0]['fidelity_matrix']))
