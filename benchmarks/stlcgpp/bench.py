"""telos vs STLCG++ (the PyTorch package, `stlcgpp`). Data layer for benchmark.ipynb.

parity()          max |telos - STLCG++| over full robustness trajectories,
                  on randomly generated formulas and traces
grid()            (T, batch) timing grid, exact + lse semantics
sweep()           dense T scaling at batch 1: wall time, peak memory,
                  all four implementations
plot(rows, ...)   the 2x2 scaling figure
cached(name, fn)  json-backed memoization under results/

Sweep cells run in fresh subprocesses: peak-memory readings stay free of
cross-cell residue and an OOM cannot poison the run.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from telos import LSE, Model, Robustness, Trace, Until, Variable, always, eventually

HERE = Path(__file__).parent
TS = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
GRID = ((64, 1), (64, 64), (256, 1), (256, 64), (1024, 1), (1024, 16), (4096, 1))
CUTOFF_S, REPS, WARMUP = 600., 10, 3

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


def _stl(name: str, recurrent: bool = False):
    import stlcgpp.formula as F
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


def _signal(values: torch.Tensor) -> torch.Tensor:
    return values.transpose(-2, -1).contiguous()


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


def translate(formula):
    """telos Formula to the equivalent STLCG++ formula."""
    import stlcgpp.formula as F
    from telos.syntax import AbstractTop, Conjunction, Disjunction, Implies, Negation
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


def parity(n: int = 20, depths: tuple = (2, 4, 6), T: int = 37, seed: int = 0) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for depth in depths:
        for _ in range(n):
            formula = random_formula(depth, rng)
            theirs = translate(formula)
            values = torch.from_numpy(rng.standard_normal((2, T)).astype(np.float32))
            trace = Trace(values, ('p', 'q'))
            signal = _signal(values[None])[0]
            def delta(algebra, **kw):
                ours = Model(algebra)(trace >> formula, return_trajectory=True)
                return float((ours - theirs(signal, **kw)).abs().max())
            rows.append({'depth': depth, 'formula': repr(formula),
                         'exact': delta(Robustness(), approx_method='true'),
                         'lse': delta(LSE(p=1., trainable=False),
                                      approx_method='logsumexp', temperature=1.0)})
    return rows


# ---------- grid (exact + lse, fwd and fwd+bwd) ----------

def grid(device: str = 'cuda') -> list[dict]:
    rows = []
    for T, B in GRID:
        values = _values(T, B).to(device)
        signal = _signal(values.cpu()).to(device)
        for formula in ('◇p', '□(p → ◇q)', 'p U q', '(p∨q) U (p∧q)'):
            for mode, algebra, kw in (
                    ('exact', Robustness(), {'approx_method': 'true'}),
                    ('lse', LSE(p=1., trainable=False), {'approx_method': 'logsumexp', 'temperature': 1.0})):
                model = Model(algebra).to(device)
                sf = _stl(formula)
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

class FallbackRobustness(Robustness):
    """Robustness without its vectorized closed forms: sequence reductions
    revert to the generic scan/fold derivations from the four primitives.
    This is the path any algebra without closed forms takes (e.g. soft,
    non-associative meets), benchmarked as `fallback`."""
    from telos.algebras.base import TensorAlgebra as _A
    running_meet, running_join = _A.running_meet, _A.running_join
    exists, forall = _A.exists, _A.forall
    span_meet, span_join = _A.span_meet, _A.span_join


def _cell(T: int, formula: str, impl: str) -> dict:
    row = {'T': T, 'formula': formula, 'impl': impl,
           'ms': None, 'peak_gb': None, 'compile_s': None, 'status': 'ok'}
    if impl in ('telos', 'fallback'):
        model = Model(Robustness() if impl == 'telos' else FallbackRobustness()).cuda()
        values = _values(T).cuda()
        def fn():
            x = values.detach().requires_grad_(True)
            model(Trace(x, ('p', 'q')) >> FORMULAS[formula]).sum().backward()
    else:
        sf = _stl(formula, recurrent=impl == 'recurrent')
        signal = _signal(_values(T)).cuda()
        def fn():
            x = signal.detach().requires_grad_(True)
            sf(x[0], approx_method='true')[0].backward()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        row['compile_s'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        warm = time.perf_counter() - t0
        if warm > CUTOFF_S:
            row.update(ms=warm * 1e3, status='cutoff')
        else:
            reps, warmup = (REPS, WARMUP) if warm < 1. else (3, 1) if warm < 10. else (1, 0)
            row.update(ms=_median_ms(fn, torch.cuda.synchronize, reps, warmup),
                       peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    except torch.cuda.OutOfMemoryError:
        row['status'] = 'oom'
    return row


def sweep(checkpoint: str | None = None) -> list[dict]:
    """checkpoint: path to dump rows after every cell (crash resilience, live preview)."""
    rows, dead = [], set()
    for T in TS:
        for formula in SWEEP_FORMULAS:
            for impl in ('telos', 'fallback', 'masked', 'recurrent'):
                if (formula, impl) in dead:
                    continue
                # fresh subprocess per cell: keeps peak-memory readings free of
                # cross-cell residue and isolates OOMs
                proc = subprocess.run(
                    [sys.executable, __file__, str(T), formula, impl],
                    capture_output=True, text=True, timeout=3600)
                out = proc.stdout.strip().splitlines()
                row = (json.loads(out[-1]) if out else
                       {'T': T, 'formula': formula, 'impl': impl, 'ms': None,
                        'peak_gb': None, 'compile_s': None, 'status': 'error'})
                if row['status'] != 'ok':
                    dead.add((formula, impl))
                rows.append(row)
                if checkpoint:
                    Path(checkpoint).write_text(json.dumps(rows, indent=1))
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


STYLE = {  # impl -> (color, linestyle); hue = tool, shade = variant (dark: fast path)
    'telos':     ('#1c5cab', 'solid'),
    'fallback':  ('#6da7ec', 'solid'),
    'masked':    ('#0d8c5c', 'solid'),
    'recurrent': ('#52c69a', 'solid'),
}
LEGEND = {'telos': 'telos (algebra-native)', 'fallback': 'telos (algebra-agnostic fallback)',
          'masked': 'STLCG++ (masked)', 'recurrent': 'STLCG++ (recurrent)'}
CRITICAL = '#d03b3b'  # OOM marker


def plot(rows: list[dict], title: str, total_gb: float | None = None):
    import math
    import matplotlib.pyplot as plt
    ink, ink2, muted, grid_c = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'

    def pts(formula, impl, key):
        sel = [(r['T'], r[key]) for r in rows
               if r['formula'] == formula and r['impl'] == impl
               and r['status'] == 'ok' and r[key] is not None]
        return zip(*sel) if sel else ((), ())

    def stop(formula, impl):
        sel = [(r['T'], r['status']) for r in rows
               if r['formula'] == formula and r['impl'] == impl and r['status'] != 'ok']
        return min(sel) if sel else None

    present = [i for i in STYLE if any(r['impl'] == i for r in rows)]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), facecolor='white')
    formulas, metrics = ('□(p → ◇q)', 'p U q'), ('ms', 'peak_gb')
    for (i, formula), (j, key) in ((f, m) for f in enumerate(formulas) for m in enumerate(metrics)):
        ax = axes[i][j]
        for impl in reversed(present):
            color, ls = STYLE[impl]
            ts, ys = pts(formula, impl, key)
            if ts:
                ax.plot(ts, ys, color=color, lw=2, ls=ls, marker='o', ms=4)
            s = stop(formula, impl)
            if s and ts and s[1] != 'oom' and key == 'ms':
                ax.annotate(f'{s[1].upper()}\nat T={s[0]}', xy=(ts[-1], ys[-1]),
                            ha='left', va='top', xytext=(8, -8),
                            textcoords='offset points', color=color, fontsize=8.5)
            if s and len(ts) > 1 and s[1] == 'oom':
                # dashed continuation of the established log-log slope; red x at the OOM point
                lt, ly = [[math.log(v) for v in vs[-4:]] for vs in (ts, ys)]
                mt, my = sum(lt) / len(lt), sum(ly) / len(ly)
                slope = (sum((a - mt) * (b - my) for a, b in zip(lt, ly))
                         / sum((a - mt) ** 2 for a in lt))
                y_est = ys[-1] * (s[0] / ts[-1]) ** slope
                ax.plot([ts[-1], s[0]], [ys[-1], y_est], color=color, lw=1.5, ls=(0, (2, 3)))
                ax.plot([s[0]], [y_est], marker='x', ms=9, mew=2.5, color=CRITICAL)
        ax.set(xscale='log', yscale='log')
        ax.set_title(f'{formula}: ' + ('wall time' if key == 'ms' else 'peak memory'),
                     color=ink, fontsize=10.5, pad=10)
        if i == 1:
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
        if j == 0:
            ax.set_ylabel('forward+backward, ms', color=ink2, fontsize=9.5)
        else:
            ax.set_ylabel('peak allocated, GB', color=ink2, fontsize=9.5)
    handles = [plt.Line2D([], [], color=STYLE[i][0], ls=STYLE[i][1], lw=2,
                          marker='o', ms=4, label=LEGEND[i]) for i in present]
    handles.append(plt.Line2D([], [], color=CRITICAL, ls='none',
                              marker='x', ms=8, mew=2.5, label='OOM'))
    leg = fig.legend(handles=handles, loc='upper left', ncol=len(handles), frameon=False,
                     fontsize=9, bbox_to_anchor=(0.01, 0.985), handlelength=2.6,
                     columnspacing=1.4)
    for text, h in zip(leg.get_texts(), handles):
        text.set_color(h.get_color())
    fig.suptitle(title, color=ink, fontsize=11, x=0.02, ha='left', y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


if __name__ == '__main__':  # subprocess cell entry: bench.py <T> <formula> <impl>
    T, formula, impl = sys.argv[1:4]
    print(json.dumps(_cell(int(T), formula, impl)))
