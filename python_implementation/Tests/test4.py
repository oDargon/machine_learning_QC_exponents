import os
import tempfile
from numpy import array, allclose
from source.exponent_handler import *

# -----------------------------
# Construct reference object
# -----------------------------

exponents = [
    # s
    array([
        9497.9344, 1416.8112, 321.45994, 91.124163, 29.999891, 11.017631,
        4.3728010, 1.8312560, .80226100, .36264800, .11399500,
        .05123700, .02246800, .00786000
    ]),
    # p
    array([
        13.119504, 3.0774242, 1.0988005, .43577840, .18024320,
        .07613330, .03254650, .01401820, .00490640
    ]),
    # d
    array([
        .45000000, .15750000, .05512500, .01929380
    ]),
    # f
    array([
        .24000000, .09600000, .03840000
    ]),
    # g
    array([
        0.128000
    ]),
]

contractions = [
    array([
        [ 0.00009669, -0.00001245],
        [ 0.00011649, -0.00010463],
        [ 0.00018111, -0.00007921],
        [ 0.00041212, -0.00030282],
        [ 0.00060871, -0.00007839],
        [ 0.00074119, -0.00067059],
        [ 0.00115621, -0.00040184],
        [ 0.00153993, -0.00010821],
        [ 0.00310152, -0.00040071],
        [ 0.00377577, -0.00339550],
        [ 0.00614057, -0.00311184],
        [ 0.01736654, -0.01796344],
        [ 0.01266905, -0.00164525],
        [ 0.01649026, -0.01530456],
    ]),
    # p (uncontracted → identity inferred later)
    array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]),
]

ref = Exponent_set(
    label="ANO-L",
    atom_name="Li",
    method="CASPT2",
    exponents=exponents,
    contractions=contractions,
    energy=-7.4333844949,
    contracted=True,
)

# -----------------------------
# Save → Load round trip
# -----------------------------

with tempfile.TemporaryDirectory() as tmp:
    path = ref.save(tmp, overwrite=True)
    assert path.endswith(".expo")
    assert os.path.exists(path)

    loaded = Exponent_set.load(path)

# -----------------------------
# Metadata asserts
# -----------------------------

assert loaded.atom_name == ref.atom_name
assert loaded.method == ref.method
# assert loaded.label == ref.label
assert allclose(loaded.energy, ref.energy)
assert loaded.contracted is True

# -----------------------------
# Structural asserts
# -----------------------------

assert len(loaded.exponents) == len(ref.exponents)
assert len(loaded.contractions) == len(ref.contractions)

for e0, e1 in zip(ref.exponents, loaded.exponents):
    assert e0.shape == e1.shape
    assert allclose(e0, e1)

for c0, c1 in zip(ref.contractions, loaded.contractions):
    assert c0.shape == c1.shape
    assert allclose(c0, c1)

# -----------------------------
# Derived quantities
# -----------------------------

for exp, cont, n_cont in zip(
    loaded.exponents,
    loaded.contractions,
    loaded.n_contracted,
):
    assert exp.shape[0] == cont.shape[0]
    assert cont.shape[1] == n_cont
