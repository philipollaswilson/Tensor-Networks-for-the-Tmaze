"""Add AdaCT-H with the language metric to the finished rdp_compare cells (same data, same test
set, same Machine score), and print the summary.
Run:  python src/rdp_summary.py
"""
import json
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rdp_envs as E  # noqa: E402
import rdp_language as RL  # noqa: E402
from rdp_compare import OUT, SEEDS, TEST, NS  # noqa: E402

METHODS = ['ours', 'language', 'adact', 'flexfringe', 'hmm', 'gru']


def add_language(path):
    r = json.loads(path.read_text())
    if 'language' in r['methods']:
        return
    env = E.ENVS[r['env']]()
    n, seed, T = r['n'], r['seed'], r['T']
    data = env.sample(n, T, 7919 * seed + n)
    Xtr = data[max(10, n // 10):]
    Xte = env.sample(TEST, T, 104729 + seed)
    t0 = time.time()
    m = RL.adact_language(Xtr, env.n_act, env.n_obs, len(env.rewards))
    loss, kept = m.logloss(Xte)
    r['methods']['language'] = dict(loss=loss, kept=kept, states=m.n_states, time=time.time() - t0,
                                    gap=loss - r['floor'])
    path.write_text(json.dumps(r))


def main():
    paths = sorted(OUT.glob('*_n*_s*.json'))
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(add_language, paths))
    gaps, states, fails = defaultdict(list), defaultdict(list), defaultdict(int)
    for p in paths:
        r = json.loads(p.read_text())
        for m, v in r['methods'].items():
            if 'gap' in v:
                gaps[r['env'], r['n'], m].append(v['gap'])
                states[r['env'], r['n'], m].append(v['states'])
            else:
                fails[r['env'], r['n'], m] += 1
    print('nats per step above the Bayes floor (timeouts or errors in brackets)')
    print(f"{'env':9s}{'n':>7s}" + ''.join(f'{m:>12s}' for m in METHODS))
    for env in E.ENVS:
        for n in NS:
            row = ''
            for m in METHODS:
                g, x = gaps[env, n, m], fails[env, n, m]
                cell = (f'{np.mean(g):.4f}' if g else '-') + (f' [{x}]' if x else '')
                row += f'{cell:>12s}'
            print(f'{env:9s}{n:7d}{row}')
    print('\nlearned states (mean); true n_pred / n_layered(H)')
    for env in E.ENVS:
        e = E.ENVS[env]()
        tr = f'{e.n_pred}/{e.n_layered(E.PAPER_H[env])}' if env != 'tiger' else f'inf/{e.n_layered(E.PAPER_H[env])}'
        for n in (NS[0], NS[2], NS[-1]):
            print(f'{env:9s}{n:7d} truth {tr:9s}' + ''.join(
                f" {m} {np.mean(states[env, n, m]):.0f}" if states[env, n, m] else f' {m} -'
                for m in ['ours', 'language', 'adact', 'flexfringe']))


if __name__ == '__main__':
    main()
