import numpy as np
from source.exponent_handler import *

A = Exponent_set()

assert A.atom_name == "X"
assert A.method == "Unknown"
assert A.exponents == []
assert A.contractions == []
assert A.energy is None
assert not A.used
assert not A.is_copy

print("Test 1 passed")


A = Exponent_set(
    label=1,
    atom_name="Be",
    exponents=[[1, 2, 3, 4], [3, 4, 5, 6]],
    method="HF"
)

assert len(A.exponents) == 2
assert isinstance(A.exponents[0], np.ndarray)
assert A.exponents[0].dtype == np.float64

# identity contractions generated
assert A.contractions[0].shape == (4, 4)
assert np.allclose(A.contractions[0], np.eye(4))
assert not A.contracted

print("Test 2 passed")



A = Exponent_set(
    atom_name="C",
    exponents=[
        [100.0, 10.0, 1.0],   # l = 0
        [5.0, 1.0],           # l = 1
        [0.8]                 # l = 2
    ]
)

assert len(A.exponents) == 3
assert A.lengths == [3, 2, 1]
assert A.n_contracted == [3, 2, 1]

print("Test 3 passed")


A = Exponent_set(
    exponents=[[1, 2, 3]],
    contractions=[[[1.0], [0.5], [0.2]]]
)

assert A.contracted
assert A.contractions[0].shape == (3, 1)
assert A.n_contracted == [1]

print("Test 4 passed")


A = Exponent_set(
    exponents=[[1, 2]],
    contractions=[[[1, 0], [0, 1]]],
    contracted=False
)

assert not A.contracted
assert A.contractions[0].shape == (2, 2)

print("Test 5 passed")


A = Exponent_set(
    label=7,
    atom_name="O",
    exponents=[[1, 2, 3]]
)

B = A.copy()

assert B.is_copy
assert B.label == A.label
assert B.atom_name == A.atom_name

B.exponents[0][0] = 999.0
assert A.exponents[0][0] != 999.0

print("Test 6 passed")


A = Exponent_set(exponents=[[1, 2]])

assert A.energy is None
assert not A.used

A.energy = -75.123456
A.used = True

assert A.energy < 0
assert A.used

print("Test 7 passed")


A = Exponent_set(
    1,
    "Be",
    [[1, 2, 3, 4], [3, 4, 5, 6], [4, 6, 7, 8]],
    method="HF"
)

print(A)

assert len(A.exponents) == 3
assert A.atom_name == "Be"
assert A.method == "HF"
assert not A.contracted

# print("Test 8 passed")



# ---- s-type (l = 0) ----
exp_s = np.array(
    [
        22628.599, 3372.3181, 760.35040, 211.74048,
        67.223468, 23.372177, 8.7213730, 3.4680910,
        1.4521440, 0.60861500, 0.25768600, 0.10417600,
        0.04242700, 0.01484900,
    ],
    dtype=np.float64,
)
cont_s = np.eye(14, dtype=np.float64)

# ---- p-type (l = 1) ----
exp_p = np.array(
    [
        33.710184, 8.0576495, 2.8364714, 1.0999657,
        0.44339640, 0.18222640, 0.07572410,
        0.03168540, 0.01108990,
    ],
    dtype=np.float64,
)
cont_p = np.eye(9, dtype=np.float64)

# ---- d-type (l = 2) ----
exp_d = np.array(
    [
        1.4000000, 0.49000000, 0.17150000, 0.06002500
    ],
    dtype=np.float64,
)
cont_d = np.eye(4, dtype=np.float64)

# ---- construct exponent set ----
A = Exponent_set(
    label=1,
    atom_name="Be",
    method="ANO-RCC",
    exponents=[exp_s, exp_p, exp_d],
    contractions=[cont_s, cont_p, cont_d],
    contracted=True,
)

# ---- sanity checks ----
assert A.atom_name == "Be"
assert A.method == "ANO-RCC"
assert A.contracted is True
assert len(A.exponents) == 3

assert A.lengths == [14, 9, 4]
assert A.n_contracted == [14, 9, 4]

for i in range(3):
    assert np.allclose(A.contractions[i], np.eye(A.lengths[i]))

for exp in A.exponents:
    assert exp.ndim == 1
    assert exp.dtype == np.float64

for cont, exp in zip(A.contractions, A.exponents):
    assert cont.shape == (exp.shape[0], exp.shape[0])
    assert cont.dtype == np.float64

print("Molcas Be ANO (real exponents) test passed\n")
print(A)
