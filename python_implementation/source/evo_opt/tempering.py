from typing import Callable
from numpy import ndarray, log, exp, array, float64, arange, column_stack
from numpy.linalg import lstsq


REGISTRY: dict[str, Callable] = {}


def register(name: str, fn: Callable[[int, int], ndarray]) -> None:
    """Register a tempering generator under a name.
    Generator signature: (n: int, m: int) -> ndarray of shape (n, m)
    where G[k, i] = g_i(k).
    """
    REGISTRY[name] = fn


class Tempering_Codec:
    """
    Codec mapping M parameters directly to N exponents via:
        ln α_k = Σ_i a_i * g_i(k)
    i.e.  α = exp(G @ a),  G[k, i] = g_i(k).

    All M parameters are free. encode() fits them from real exponents via lstsq.
    decode() with an optional n argument allows generating a different number of
    exponents from the same params (extension).
    """

    def __init__(self, generator: Callable[[int, int], ndarray], name: str, m: int, n: int):
        self.generator = generator
        self.name      = name
        self.m         = m
        self.n         = n

    def decode(self, params: ndarray, n: int | None = None) -> ndarray:
        n = n if n is not None else self.n
        G = self.generator(n, self.m)
        return exp(G @ array(params, dtype=float64))

    def encode(self, exponents: ndarray) -> ndarray:
        n          = len(exponents)
        G          = self.generator(n, self.m)
        a, _, _, _ = lstsq(G, log(array(exponents, dtype=float64)), rcond=None)
        return a


def from_registry(name: str, m: int, n: int) -> Tempering_Codec:
    if name not in REGISTRY:
        raise KeyError(f"No tempering generator '{name}' registered")
    return Tempering_Codec(REGISTRY[name], name, m, n)


# --- built-in generators ---

def _polynomial(n: int, m: int) -> ndarray:
    k = arange(n, dtype=float64)
    return column_stack([k**i for i in range(m)])

register("polynomial", _polynomial)
