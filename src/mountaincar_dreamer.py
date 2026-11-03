"""DreamerV3 (official JAX code, lib/dreamerv3) on position-only Mountain Car.

The environment is mountaincar_env.MountainCarPO wrapped in DreamerV3's embodied API:
the same observation symbol (binned position + reward bit, as a categorical token), the
same reward, start states (one env, same seed), 200-step episodes and flag schedule.
The episode never terminates (is_terminal stays False; the time limit ends it), as for
our agents. Each episode's return, start and flag are logged from inside the env to
<logdir>/episodes.jsonl, so scoring matches mountaincar_env.run_agent; the eval env
cycles through EVAL_STARTS with the policy in eval mode.

Run with the dreamer conda env (Python 3.11, jax 0.4.33 CPU), from the repo root:
  C:/Users/darth/miniconda3/envs/dreamer/python.exe src/mountaincar_dreamer.py \
      --seed 0 --episodes 500 [--moving]
Model size1m (the smallest preset), float32 on CPU, train_ratio 512.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'lib' / 'dreamerv3'))

import elements  # noqa: E402
import embodied  # noqa: E402
import numpy as np  # noqa: E402

import mountaincar_env as E  # noqa: E402


class MountainCarEmbodied(embodied.Env):

    def __init__(self, task, seed=0, moving=False, logdir=None, evaluate=False):
        self.evaluate, self.moving = evaluate, moving
        self.env = E.MountainCarPO(seed + (10_000 if evaluate else 0))
        self.log = pathlib.Path(logdir) / ('eval_episodes.jsonl' if evaluate else 'episodes.jsonl')
        self.episode, self.t, self.ret = -1, E.T_MAX, 0.0
        self.train_episode = 0

    @property
    def obs_space(self):
        return {
            'token': elements.Space(np.int32, (), 0, E.NO),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {'reset': elements.Space(bool), 'action': elements.Space(np.int32, (), 0, E.NA)}

    def train_episodes(self):
        f = self.log.with_name('episodes.jsonl')
        return sum(1 for _ in open(f)) if f.exists() else 0

    def _flag(self):
        # eval follows the training env's flag, read from its episode file
        n = self.train_episodes() if self.evaluate else self.episode
        return E.flag_schedule(self.moving, n)

    def step(self, action):
        if action.pop('reset') or self.t >= E.T_MAX:
            self.episode += 1
            self.flag = self._flag()
            x0 = E.EVAL_STARTS[self.episode % len(E.EVAL_STARTS)] if self.evaluate else None
            o = self.env.reset(self.flag, x0)
            self.x0, self.t, self.ret = self.env.x, 0, 0.0
            return self._obs(o, 0.0, is_first=True)
        o, r = self.env.step(int(action['action']))
        self.t += 1
        self.ret += r
        last = self.t >= E.T_MAX
        if last:
            rec = {'episode': self.episode, 'flag': self.flag, 'return': self.ret,
                   'x0': self.x0, 'optimal': E.oracle_return(self.x0, self.flag),
                   'train_episodes': self.train_episodes()}
            with open(self.log, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
        return self._obs(o, r, is_last=last)

    def _obs(self, o, r, is_first=False, is_last=False):
        return dict(token=np.int32(o), reward=np.float32(r), is_first=is_first,
                    is_last=is_last, is_terminal=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--moving', action='store_true')
    ap.add_argument('--train_ratio', type=float, default=512)
    args = ap.parse_args()
    from dreamerv3 import main as dm

    cond = 'mountaincar_moving' if args.moving else 'mountaincar'
    logdir = REPO / 'results' / cond / f'dreamerv3_s{args.seed}'
    logdir.mkdir(parents=True, exist_ok=True)
    for f in ('episodes.jsonl', 'eval_episodes.jsonl'):
        (logdir / f).unlink(missing_ok=True)

    built = []          # run.debug (the default) builds envs in-process: spaces, train, eval

    def make_env(config, index, **overrides):
        built.append(index)
        env = MountainCarEmbodied('mc', seed=args.seed, moving=args.moving, logdir=logdir,
                                  evaluate=len(built) >= 3)
        return dm.wrap_env(env, config)
    dm.make_env = make_env

    dm.main([
        '--configs', 'size1m',
        '--task', 'dummy_mc', '--seed', str(args.seed), '--logdir', str(logdir / 'dreamer'),
        '--script', 'train_eval',
        '--run.steps', str(args.episodes * E.T_MAX), '--run.envs', '1', '--run.eval_envs', '1',
        '--run.train_ratio', str(args.train_ratio), '--run.save_every', '1e9',
        '--run.eval_eps', str(len(E.EVAL_STARTS)), '--run.report_every', '300',
        '--run.usage.nvsmi', 'False',
        '--jax.platform', 'cpu', '--jax.compute_dtype', 'float32', '--jax.prealloc', 'False',
        '--logger.outputs', 'jsonl',
    ])


if __name__ == '__main__':
    main()
