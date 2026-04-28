# Telos

[![tests](https://github.com/konstantinosKokos/telos/actions/workflows/tests.yml/badge.svg)](https://github.com/konstantinosKokos/telos/actions/workflows/tests.yml)

_A framework for evaluating and back-propagating through linear temporal logic traces, in pytorch._

## About

Telos is a pytorch library that aims to make LTL easier to use in downstream ML tasks (i.e., soft trace verification or 
logic-based loss conditioning). The differentiator from related contemporary frameworks is a strict and precise separation 
of concerns. Instead of a monolithic back-end, Telos opts for an explicit distinction between syntax, semantics 
and their interface. The result is a clean contract that ensures plug-and-play support for arbitrary algebras, 
cross-algebra comparison, and dynamic switching. With Telos, one can use a differentiable algebra during training 
and a boolean algebra during validation with little ceremony.

## Project Structure

### Syntax

`telos.syntax` defines the shapes of LTL formulas, following standard textbook conventions for primitive operations and 
using python class inheritance to get away with the language's lack of support for ADTs. 

The grammar is defined as
```
Φ := A          -- atomic variable
    | ⊤         -- top
    | ⊥         -- bottom
    | Φ₁ → Φ₂   -- material implication
    | Φ₁ ∨ Φ₂   -- disjunction
    | Φ₁ ∧ Φ₂   -- conjunction
    | ¬Φ        -- negation
    | X(Φ)      -- temporal next
    | U(Φ₁, Φ₂) -- temporal until
    | ◇Φ        -- eventually
    | □Φ        -- always
    | Φ₁ ↔ Φ₂   -- if and only if
```

The operators below are treated as primitives:
- `A` -- `Variable(A)`, where `A` is any valid python identifier
- `⊤` -- `AbstractTop()`
- `⊥` -- `AbstractBottom()`
- `Φ₁ → Φ₂` -- `Implies(l, r)`, shorthand `l > r`
- `Φ₁ ∨ Φ₂` -- `Disjunction(l, r)`, shorthand `l | r`
- `Φ₁ ∧ Φ₂` -- `Conjunction(l, r)`, shorthand `l & r`
- `¬Φ` -- `Negation(x)`, shorthand `~x`
- `X(Φ)` -- `Next(x)`
- `U(Φ₁, Φ₂)` -- `Until(l, r)`

The rest are defined as composite functions:
- `◇Φ` -- `eventually(x)`, defined as `U(⊤, Φ)`
- `□Φ` -- `always(x)`, defined as `¬◇¬Φ`
- `Φ₁ ↔ Φ₂` -- `iff(l, r)`, defined as `(Φ₁ → Φ₂) ∧ (Φ₂ → Φ₁)`

### Semantics

`telos.algebras` defines the interpretations under which formulas are evaluated. An algebra fixes how each 
propositional connective is computed on values from a chosen carrier; the temporal operators are built on top of these 
via batch-vectorized sequence reductions. The module ships several standard algebras and exposes an abstract interface 
for adding more.

#### Abstract

An `Algebra` is a `torch.nn.Module` that fixes the semantics on a carrier set. It declares two designated elements 
(`top` and `bottom`, registered as buffers) and four pointwise primitives:

| Connective   | Method          |
|--------------|-----------------|
| `Φ₁ ∧ Φ₂`    | `meet(x, y)`    |
| `Φ₁ ∨ Φ₂`    | `join(x, y)`    |
| `Φ₁ → Φ₂`    | `implies(x, y)` |
| `¬Φ`         | `neg(x)`        |

Sequence reductions (`exists`, `forall`, `running_meet`, `running_join`, `span_meet`, `span_join`) are derived from the 
primitives via functional iterators (`scan`, `fold`, and a triangular-mask `span` combinator). Each can be overridden 
with a vectorized closed form when one exists.

Since `Algebra` extends `Module`, its parameters (if any) are first-class, and can optionally be trained end-to-end 
just like a standard pytorch module.

A convenience subclass `FuzzyBase` handles the common case of a `[0,1]` carrier with `neg(x) = 1 - x`.

#### Existing Implementations

The algebras in the table below are implemented and property tested.

**Legend**:
* idempotence (Idem)
* absorption (Abs)
* distributivity (Dist)
* complementarity (Comp). 

See `algebras` for the implementations, and `tests/test_properties.py` for the checks.

| Algebra          | Carrier    | Diff  | Train | Idem  | Abs   | Dist  | Comp  |
|------------------|------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| `Boolean`        | `𝔹`        |       |       |   ✓   |   ✓   |   ✓   |   ✓   |
| `Goedel`         | `[0, 1]`   | [^a]  |       |   ✓   |   ✓   |   ✓   |       |
| `Lukasiewicz`    | `[0, 1]`   |   ✓   |       |       |       |       |   ✓   |
| `Product`        | `[0, 1]`   |   ✓   |       |       |       |       |       |
| `Robustness`     | `ℝ ∪ {±∞}` |   ✓   |       |   ✓   |   ✓   |   ✓   |       |
| `Frank`          | `[0, 1]`   |   ✓   |   ✓   | [^c]  | [^c]  | [^c]  | [^b]  |
| `Hamacher`       | `[0, 1]`   |   ✓   |   ✓   |       |       |       |       |
| `Yager`          | `[0, 1]`   |   ✓   |   ✓   | [^b]  | [^b]  | [^b]  | [^d]  |
| `SchweizerSklar` | `[0, 1]`   |   ✓   |   ✓   |       |       |       | [^e]  |
| `AczelAlsina`    | `[0, 1]`   |   ✓   |   ✓   | [^b]  | [^b]  | [^b]  |       |
| `Dombi`          | `[0, 1]`   |   ✓   |   ✓   | [^b]  | [^b]  | [^b]  |       |
| `SugenoWeber`    | `[0, 1]`   |   ✓   |   ✓   |       |       |       | [^c]  |
| `LSE`            | `ℝ ∪ {±∞}` |   ✓   |   ✓   | [^b]  | [^b]  | [^b]  |       |

[^a]: `Implies` is _not_ differentiable in its first argument.
[^b]: When `p → ∞`.
[^c]: When `p → 0`.
[^d]: When `p = 1`.
[^e]: When `p ≥ 1`.


#### Writing your Own

Subclass `Algebra` (or `FuzzyBase`) and implement or inherit the top and bottom elements and the four 
pointwise primitives. Sequence reductions inherit defaults; override them where a vectorized closed form exists.

See `telos.algebras.goedel` for a minimal reference and `telos.algebras.frank` for an example with a trainable parameter.

### Interface

`telos.deduction` connects syntax and semantics. The three objects of interest are:
- the `Trace(values, names)` -- a glorified tensor of shape `(..., vars, time)`, naming its coordinates along `dim=-2`. Most cleanly built via the helper `mkTrace(**vars)`. 
- `Judgement(trace, conclusion)` -- a pairing of a trace and a formula whose leaf variables must occur in the trace.
- `Model(algebra)` -- a `torch.nn.Module` that lifts an algebra to a judgement evaluation engine.

Evaluation is structural recursion: pointwise constructors call the algebra's primitives.

#### Example

Here's a minimal example demonstrating an end-to-end pipeline.

```python
import torch
from telos import mkTrace, Variable, eventually, always, Model, Lukasiewicz, Product, Boolean

p, q = Variable('p'), Variable('q')              # p and q are abstract symbols
phi = always(p > eventually(q))                  # φ := □(p → ◇q)

trace_f = mkTrace(
    p=(tp := torch.rand(4, requires_grad=True)), # the fp32 progress of p (e.g., sensor measurements, classifier outputs, etc.)
    q=(tq := torch.rand(4, requires_grad=True)), # ditto for q
)                                                # the two variables packed up in a single trace
trace_b = trace_f.bool()                         # the discretization of the above trace

judgement_f = trace_f >> phi                     # the judgement τ ⊢ φ, in the fp32 domain
judgement_b = trace_b >> phi                     # the same judgement in the boolean domain

score_l = Model(Lukasiewicz())(judgement_f)      # apply the Lukasiewicz algebra on the judgement
score_p = Model(Product())(judgement_f)          # ditto, now with the Product algebra
score_b = Model(Boolean())(judgement_b)          # ditto, now with the Boolean algebra -- note the domain change

traj_l = Model(Lukasiewicz())(
    judgement_f, 
    return_trajectory=True
)                                                # return the valuation of φ at each step

goal_f  = torch.rand(())                         # some goal valuation to train on
loss    = torch.nn.functional.l1_loss(
    input=score_l,
    target=goal_f
)
loss.backward()                                  # backprop through the algebra, populating grads for tp and tq
```

Same formula, multiple algebras and evaluations across two trace dtypes.

## Implementation Notes
- Telos evaluates LTL over finite, fixed-duration traces and cannot work with infinite streams.
- `X(Φ)` is padded with `algebra.bottom` past the last time step, biasing trace-edge readings toward dissatisfaction.
- Every algebra is associated with its own domain; you're responsible for using the right dtype.