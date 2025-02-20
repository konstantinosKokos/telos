from __future__ import annotations
from abc import ABC


class Formula(ABC):
    def __repr__(self) -> str:
        return formula_repr(self)

    def __eq__(self, other) -> bool:
        return formula_eq(self, other)

    def variables(self) -> set[Variable]:
        return free(self)

    def __hash__(self) -> int:
        return formula_hash(self)


class AbstractTop(Formula):
    ...


class AbstractBottom(Formula):
    ...


class Variable(Formula):
    __match_args__ = ('name',)

    def __init__(self, name: str):
        assert len(name) == 1 and 'a' <= name <= 'z', "Only lowercase latin characters allowed as variable names."
        self.name = name


class Negation(Formula):
    __match_args__ = ('content',)

    def __init__(self, content: Formula):
        self.content = content


class Next(Formula):
    __match_args__ = ('content',)

    def __init__(self, content: Formula):
        self.content = content


class Disjunction(Formula):
    __match_args__ = ('left', 'right')

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right


class Conjunction(Formula):
    __match_args__ = ('left', 'right')

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right


class Implies(Formula):
    __match_args__ = ('left', 'right')

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right


class Until(Formula):
    __match_args__ = ('left', 'right')

    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right


def eventually(f: Formula) -> Formula:
    return Until(AbstractTop(), f)

def always(f: Formula) -> Formula:
    return Negation(eventually(Negation(f)))


def iff(left: Formula, right: Formula) -> Formula:
    return Conjunction(Implies(left, right), Implies(right, left))


def formula_repr(f: Formula) -> str:
    def par(g: Formula) -> str:
        s = formula_repr(g)
        return s if isinstance(g, (Variable, Next, Until)) else f'({s})'

    match f:
        case AbstractTop(): return '⊤'
        case AbstractBottom(): return '⊥'
        case Variable(x): return f'{x}'
        case Negation(x): return f'¬ {par(x)}'
        case Next(x): return f'𝓧 {par(x)}'
        case Disjunction(l, r): return f'{par(l)} ∨ {par(r)}'
        case Conjunction(l, r): return f'{par(l)} ∧ {par(r)}'
        case Implies(l, r): return f'{par(l)} → {(par(r))}'
        case Until(l, r): return f'{par(l)} 𝒰 {par(r)}'
        case _: raise ValueError


def formula_eq(left: Formula, right: Formula) -> bool:
    match left, right:
        case AbstractTop(), AbstractTop(): return True
        case AbstractBottom(), AbstractBottom(): return True
        case Variable(x), Variable(y): return x == y
        case Next(x), Next(y): return formula_eq(x, y)
        case Disjunction(l1, r1), Disjunction(l2, r2): return all(map(formula_eq, (l1, r1), (l2, r2)))
        case Conjunction(l1, r1), Conjunction(l2, r2): return all(map(formula_eq, (l1, r1), (l2, r2)))
        case Implies(l1, r1), Implies(l2, r2): return all(map(formula_eq, (l1, r1), (l2, r2)))
        case Until(l1, r1), Until(l2, r2): return all(map(formula_eq, (l1, r1), (l2, r2)))
        case _: return False


def free(f: Formula) -> set[Variable]:
    match f:
        case AbstractTop() | AbstractBottom(): return set()
        case Variable(x): return {Variable(x)}
        case Negation(x) | Next(x): return free(x)
        case Disjunction(l, r) | Until(l, r) | Conjunction(l, r) | Implies(l, r): return free(l) | free(r)
        case _: raise ValueError


def formula_hash(f: Formula) -> int:
    match f:
        case AbstractTop(): return hash(True)
        case AbstractBottom(): return hash(False)
        case Variable(x): return hash(x)
        case Negation(x): return hash(('n', formula_hash(x)))
        case Next(x): return hash(('x', formula_hash(x)))
        case Disjunction(l, r): return hash(('d', formula_hash(l), formula_hash(r)))
        case Conjunction(l, r): return hash(('c', formula_hash(l), formula_hash(r)))
        case Implies(l, r): return hash(('i', formula_hash(l), formula_hash(r)))
        case Until(l, r): return hash(('u', formula_hash(l), formula_hash(r)))
        case _: raise ValueError
