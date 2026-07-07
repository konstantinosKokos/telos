"""telos vs STLCG++ (stlcgpp: PyTorch, stljax: JAX). Data layer for benchmark.ipynb.

parity(backend)        max |telos - STLCG++| over full robustness trajectories
grid()                 (T, batch) timing grid, telos vs stlcgpp, exact + lse
sweep(backend)         dense T scaling at batch 1: wall time, peak memory
plot(rows, ...)        the three-panel scaling figure
cached(name, fn)       json-backed memoization under results/

JAX sweep cells run in subprocesses: jax's peak-memory counter is monotone
per process and an OOM can poison the runtime. Compile time is recorded but
excluded from timings; the cutoff applies to steady-state runtime.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import torch
from telos import LSE, Model, Robustness, Trace, Until, Variable, always, eventually

HERE = Path(__file__).parent
TS = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
GRID = ((64, 1), (64, 64), (256, 1), (256, 64), (1024, 1), (1024, 16), (4096, 1))
CUTOFF_S, REPS, WARMUP = 30., 10, 3

P, Q = Variable('p'), Variable('q')
FORMULAS = {
    '◇p': eventually(P),
    '□p': always(P),
    'p U q': Until(P, Q),
    '□(p → ◇q)': always(P > eventually(Q)),
    '◇(p ∧ □q)': eventually(P & always(Q)),
    '(p∨q) U (p∧q)': Until(P | Q, P & Q),
    '¬◇¬p': ~eventually(~P),
}
SWEEP_FORMULAS = ('□(p → ◇q)', 'p U q')


def _stl(backend: str, name: str, recurrent: bool = False):
    F = __import__('stljax.formula' if backend == 'jax' else 'stlcgpp.formula',
                   fromlist=['formula'])
    sp = F.GreaterThan(F.Predicate('p', lambda s: s[..., 0]), 0.0)
    sq = F.GreaterThan(F.Predicate('q', lambda s: s[..., 1]), 0.0)
    ev, alw, until = ((F.EventuallyRecurrent, F.AlwaysRecurrent, F.UntilRecurrent)
                      if recurrent else (F.Eventually, F.Always, F.Until))
    return {
        '◇p': ev(sp),
        '□p': alw(sp),
        'p U q': until(sp, sq),
        '□(p → ◇q)': alw(F.Implies(sp, ev(sq))),
        '◇(p ∧ □q)': ev(F.And(sp, alw(sq))),
        '(p∨q) U (p∧q)': until(F.Or(sp, sq), F.And(sp, sq)),
        '¬◇¬p': F.Negation(ev(F.Negation(sp))),
    }[name]


def _values(T: int, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(T * 1000 + batch)
    return torch.randn(batch, 2, T, generator=g)


def _signal(backend: str, values: torch.Tensor):
    arr = values.transpose(-2, -1).contiguous()
    if backend == 'jax':
        import jax
        return jax.device_put(arr.numpy())
    return arr


def _median_ms(fn, sync, reps: int = REPS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    sync()
    def once():
        t0 = time.perf_counter()
        fn()
        sync()
        return time.perf_counter() - t0
    return sorted(once() for _ in range(reps))[reps // 2] * 1e3


# ---------- parity ----------

def random_formula(depth: int, rng):
    from telos.syntax import Conjunction, Disjunction, Implies, Negation
    if depth == 0:
        return P if rng.random() < .5 else Q
    op = rng.choice(('not', 'eventually', 'always', 'and', 'or', 'implies', 'until'))
    deep = lambda: random_formula(depth - 1, rng)
    sub = lambda: random_formula(int(rng.integers(0, depth)), rng)
    return {'not': lambda: Negation(deep()),
            'eventually': lambda: eventually(deep()),
            'always': lambda: always(deep()),
            'and': lambda: Conjunction(deep(), sub()),
            'or': lambda: Disjunction(deep(), sub()),
            'implies': lambda: Implies(deep(), sub()),
            'until': lambda: Until(deep(), sub())}[op]()


def translate(formula, backend: str):
    """telos Formula to the equivalent STLCG++ formula."""
    from telos.syntax import AbstractTop, Conjunction, Disjunction, Implies, Negation
    F = __import__('stljax.formula' if backend == 'jax' else 'stlcgpp.formula',
                   fromlist=['formula'])
    atom = {v: F.GreaterThan(F.Predicate(v, lambda s, i=i: s[..., i]), 0.0)
            for i, v in enumerate(('p', 'q'))}
    def go(phi):
        match phi:
            case Variable(name): return atom[name]
            case Until(AbstractTop(), x): return F.Eventually(go(x))
            case Negation(x): return F.Negation(go(x))
            case Conjunction(l, r): return F.And(go(l), go(r))
            case Disjunction(l, r): return F.Or(go(l), go(r))
            case Implies(l, r): return F.Implies(go(l), go(r))
            case Until(l, r): return F.Until(go(l), go(r))
            case _: raise ValueError(phi)
    return go(formula)


def parity(backend: str, n: int = 20, depths: tuple = (2, 4, 6), T: int = 37,
           seed: int = 0) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for depth in depths:
        for _ in range(n):
            formula = random_formula(depth, rng)
            theirs = translate(formula, backend)
            values = torch.from_numpy(rng.standard_normal((2, T)).astype(np.float32))
            trace = Trace(values, ('p', 'q'))
            signal = _signal(backend, values[None])[0]
            def delta(algebra, **kw):
                ours = Model(algebra)(trace >> formula, return_trajectory=True)
                ref = torch.asarray(theirs(signal, **kw).__array__(), copy=True)
                return float((ours - ref).abs().max())
            rows.append({'depth': depth, 'formula': repr(formula),
                         'exact': delta(Robustness(), approx_method='true'),
                         'lse': delta(LSE(p=1., trainable=False),
                                      approx_method='logsumexp', temperature=1.0)})
    return rows


# ---------- grid (telos vs stlcgpp, exact + lse, fwd and fwd+bwd) ----------

def grid(device: str = 'cuda') -> list[dict]:
    rows = []
    for T, B in GRID:
        values = _values(T, B).to(device)
        signal = _signal('torch', values.cpu()).to(device)
        for formula in ('◇p', '□(p → ◇q)', 'p U q', '(p∨q) U (p∧q)'):
            for mode, algebra, kw in (
                    ('exact', Robustness(), {'approx_method': 'true'}),
                    ('lse', LSE(p=1., trainable=False), {'approx_method': 'logsumexp', 'temperature': 1.0})):
                model = Model(algebra).to(device)
                sf = _stl('torch', formula)
                one = lambda s: sf(s, **kw)
                s_fn = torch.vmap(one) if B > 1 else lambda s: one(s[0])[None]
                for backward in (False, True):
                    def t_run():
                        x = values.detach().requires_grad_(backward)
                        out = model(Trace(x, ('p', 'q')) >> FORMULAS[formula])
                        if backward:
                            out.sum().backward()
                    def s_run():
                        x = signal.detach().requires_grad_(backward)
                        out = s_fn(x)[:, 0]
                        if backward:
                            out.sum().backward()
                    sync = torch.cuda.synchronize if device == 'cuda' else (lambda: None)
                    row = {'T': T, 'B': B, 'formula': formula, 'mode': mode,
                           'pass': 'fwd+bwd' if backward else 'fwd'}
                    for key, fn in (('telos_ms', t_run), ('stlcgpp_ms', s_run)):
                        try:
                            row[key] = _median_ms(fn, sync, reps=20, warmup=5)
                        except torch.cuda.OutOfMemoryError:
                            row[key] = None
                            torch.cuda.empty_cache()
                    rows.append(row)
    return rows


# ---------- sweep (dense T, batch 1, exact, fwd+bwd, time + peak memory) ----------

def _cell(backend: str, T: int, formula: str, impl: str) -> dict:
    row = {'T': T, 'formula': formula, 'impl': impl,
           'ms': None, 'peak_gb': None, 'compile_s': None, 'status': 'ok'}
    if impl == 'telos':
        model = Model(Robustness()).cuda()
        values = _values(T).cuda()
        def fn():
            x = values.detach().requires_grad_(True)
            model(Trace(x, ('p', 'q')) >> FORMULAS[formula]).sum().backward()
        sync, oom = torch.cuda.synchronize, (torch.cuda.OutOfMemoryError,)
        peak = lambda: torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    else:
        import jax
        sf = _stl(backend, formula, recurrent=impl == 'recurrent')
        signal = _signal(backend, _values(T)) if backend == 'jax' else _signal('torch', _values(T)).cuda()
        if backend == 'jax':
            vg = jax.jit(jax.value_and_grad(lambda s: sf(s[0], approx_method='true')[0]))
            def fn():
                vg(signal)[1].block_until_ready()
            sync, oom = (lambda: None), (jax.errors.JaxRuntimeError, RuntimeError)
            peak = lambda: (jax.local_devices()[0].memory_stats() or {}).get('peak_bytes_in_use', 0) / 1e9
        else:
            def fn():
                x = signal.detach().requires_grad_(True)
                sf(x[0], approx_method='true')[0].backward()
            sync, oom = torch.cuda.synchronize, (torch.cuda.OutOfMemoryError,)
            peak = lambda: torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        fn()
        sync()
        row['compile_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn()
        sync()
        warm = time.perf_counter() - t0
        if warm > CUTOFF_S:
            row.update(ms=warm * 1e3, status='cutoff')
        else:
            reps, warmup = (REPS, WARMUP) if warm < 1. else (3, 1)
            row.update(ms=_median_ms(fn, sync, reps, warmup), peak_gb=peak())
    except oom as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or 'RESOURCE_EXHAUSTED' in str(e):
            row['status'] = 'oom'
        else:
            raise
    return row


def sweep(backend: str) -> list[dict]:
    rows, dead = [], set()
    for T in TS:
        for formula in SWEEP_FORMULAS:
            for impl in ('telos', 'masked', 'recurrent'):
                if (formula, impl) in dead:
                    continue
                if backend == 'jax' and impl != 'telos':
                    proc = subprocess.run(
                        [sys.executable, __file__, backend, str(T), formula, impl],
                        capture_output=True, text=True, timeout=600)
                    out = proc.stdout.strip().splitlines()
                    row = (json.loads(out[-1]) if out else
                           {'T': T, 'formula': formula, 'impl': impl, 'ms': None,
                            'peak_gb': None, 'compile_s': None,
                            'status': 'oom' if 'RESOURCE_EXHAUSTED' in proc.stderr else 'error'})
                else:
                    row = _cell(backend, T, formula, impl)
                    torch.cuda.empty_cache()
                if row['status'] != 'ok':
                    dead.add((formula, impl))
                rows.append(row)
    return rows


# ---------- persistence and plotting ----------

def cached(name: str, fn) -> list[dict]:
    path = HERE / 'results' / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    rows = fn()
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=1))
    return rows


def speedups(rows: list[dict]) -> dict:
    from math import exp, log
    gm = lambda xs: exp(sum(map(log, xs)) / len(xs))
    out = {}
    for kind, test in (('◇/□', lambda f: 'U' not in f), ('U', lambda f: 'U' in f)):
        for pass_ in ('fwd', 'fwd+bwd'):
            xs = [r['stlcgpp_ms'] / r['telos_ms'] for r in rows
                  if test(r['formula']) and r['pass'] == pass_ and r['stlcgpp_ms']]
            out[kind, pass_] = gm(xs), min(xs), max(xs)
    out['cells'] = len(rows)
    out['oom'] = sorted({(r['T'], r['B']) for r in rows if r['stlcgpp_ms'] is None})
    return out


def plot(rows: list[dict], title: str, total_gb: float | None = None):
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid_c = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
    colors = {'telos': '#2a78d6', 'masked': '#1baf7a', 'recurrent': '#eda100'}

    def pts(formula, impl, key):
        sel = [(r['T'], r[key]) for r in rows
               if r['formula'] == formula and r['impl'] == impl
               and r['status'] == 'ok' and r[key] is not None]
        return zip(*sel) if sel else ((), ())

    def stop(formula, impl):
        sel = [(r['T'], r['status']) for r in rows
               if r['formula'] == formula and r['impl'] == impl and r['status'] != 'ok']
        return min(sel) if sel else None

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), facecolor='white')
    panels = (('□(p → ◇q)', 'ms'), ('p U q', 'ms'), ('p U q', 'peak_gb'))
    for ax, (formula, key) in zip(axes, panels):
        for impl in ('recurrent', 'masked', 'telos'):
            ts, ys = pts(formula, impl, key)
            if ts:
                ax.plot(ts, ys, color=colors[impl], lw=2, marker='o', ms=4.5)
                ax.annotate(impl, xy=(ts[-1], ys[-1]), xytext=(6, 0),
                            textcoords='offset points', color=colors[impl],
                            fontsize=9.5, fontweight='bold',
                            va='bottom' if impl != 'telos' else 'center')
            if (s := stop(formula, impl)) and key == 'ms':
                ax.axvline(s[0], color=colors[impl], lw=1, ls=(0, (2, 3)))
                if ts:
                    ax.annotate(f'{s[1].upper()}\nat T={s[0]}', xy=(ts[-1], ys[-1]),
                                ha='left', va='top', xytext=(8, -8),
                                textcoords='offset points', color=colors[impl], fontsize=8.5)
        if key == 'peak_gb' and total_gb:
            ax.axhline(total_gb, color=muted, lw=1, ls=(0, (4, 3)))
            ax.annotate(f'GPU capacity ({total_gb:.0f} GB)', xy=(64, total_gb),
                        xytext=(0, 5), textcoords='offset points', color=ink2, fontsize=8.5)
        ax.set(xscale='log', yscale='log')
        ax.set_title(f'{formula}: ' + ('wall time' if key == 'ms' else 'peak memory'),
                     color=ink, fontsize=10.5, pad=10)
        ax.set_xlabel('trace length T', color=ink2, fontsize=9.5)
        ax.set_xticks([64, 256, 1024, 4096], ['64', '256', '1024', '4096'])
        ax.minorticks_off()
        ax.grid(True, which='major', color=grid_c, lw=0.75)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid_c)
        ax.margins(x=0.15)
    axes[0].set_ylabel('forward+backward, ms', color=ink2, fontsize=9.5)
    axes[2].set_ylabel('peak allocated, GB', color=ink2, fontsize=9.5)
    fig.suptitle(title, color=ink, fontsize=11, x=0.02, ha='left', y=1.02)
    fig.tight_layout()
    return fig


if __name__ == '__main__':  # subprocess cell entry: bench.py <backend> <T> <formula> <impl>
    backend, T, formula, impl = sys.argv[1:5]
    print(json.dumps(_cell(backend, int(T), formula, impl)))
