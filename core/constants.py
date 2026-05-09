"""
MONAD Framework — Core Constants
=================================
All constants derive from the Pi-Pentad identity: π = 5·arccos(φ/2)

Fundamental morphemes: {∅, π, φ, i}  where i = sqrt(φψ) = sqrt(-1)
Everything else is derived.

All assertions are verified at import time. If any assertion fires, the
numerical environment is broken in a way that would corrupt the framework.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Morphemic generators
# ---------------------------------------------------------------------------

PHI: float = (1.0 + np.sqrt(5.0)) / 2.0   # Golden ratio  ≈ 1.6180339887
PSI: float = (1.0 - np.sqrt(5.0)) / 2.0   # Golden conjugate = -1/φ ≈ -0.6180

PI: float = np.pi                           # π ≈ 3.14159265358979

# Fundamental root identity: φψ = -1  (Fibonacci-matrix determinant)
assert abs(PHI * PSI + 1.0) < 1e-14, "Root identity φψ=-1 failed"

# Pi-Pentad identity: π = 5·arccos(φ/2)  [exact, Euclidean — Euclid Book IV]
assert abs(PI - 5.0 * np.arccos(PHI / 2.0)) < 1e-10, "Pi-Pentad π=5·arccos(φ/2) failed"

# ---------------------------------------------------------------------------
# Derived φ-powers
# ---------------------------------------------------------------------------

PHI_SQ:    float = PHI ** 2       # φ² = φ+1     ≈ 2.6180  (also Hopf saturation)
PHI_INV:   float = 1.0 / PHI     # φ⁻¹ = φ-1    ≈ 0.6180  (also = PSI + 1)
PHI_INV2:  float = PHI ** (-2)   # φ⁻²           ≈ 0.3820
PHI_INV3:  float = PHI ** (-3)   # φ⁻³           ≈ 0.2361
PHI_INV5:  float = PHI ** (-5)   # φ⁻⁵           ≈ 0.0902
PHI_INV9:  float = PHI ** (-9)   # φ⁻⁹           ≈ 0.0132 (Top Yukawa correction)
PHI_INV10: float = PHI ** (-10)  # φ⁻¹⁰          ≈ 0.0081

# MONAD identity: φ⁻⁵ + 5φ⁻² = 2  [proved symbolically, exact]
# Proof: (5φ-8) + 5(2-φ) = 5φ-8+10-5φ = 2  ✓
assert abs(PHI_INV5 + 5.0 * PHI_INV2 - 2.0) < 1e-12, "MONAD identity φ⁻⁵+5φ⁻²=2 failed"

# ---------------------------------------------------------------------------
# Cosmological / rendering split
# ---------------------------------------------------------------------------

RENDERING_FRACTION: float = PHI_INV5 / 2.0        # ≈ 0.04508 = 4.51%  observable sector
DARK_FRACTION:      float = 5.0 * PHI_INV2 / 2.0  # ≈ 0.95492 = 95.49% implicate sector

assert abs(RENDERING_FRACTION + DARK_FRACTION - 1.0) < 1e-12, "Rendering split must sum to 1"

# ---------------------------------------------------------------------------
# Feigenbaum constants (MONAD versions)
# ---------------------------------------------------------------------------

# Toroidal δ — matches classical Feigenbaum δ ≈ 4.66920 to 3.2 ppm
# Physical interpretation: aether drag = structural residual, not numerical error
DELTA_F: float = 2.0 * PI - PHI + PHI_INV10 / 2.0   # ≈ 4.66922

# Morphemic α — classical Feigenbaum α ≈ 2.50290  [sign corrected May 2026]
ALPHA_F: float = PI - PHI_INV - PHI_INV5 / 4.0       # ≈ 2.50102

# Dual-class Feigenbaum limits
# δ_L: observer-free geometric limit (purely transcendental: π + arctan(e^π))
# δ_S: observer-entangled algebraic limit (toroidal: 2π - φ + φ⁻¹⁰/2)
DELTA_L: float = PI + np.arctan(np.exp(PI))           # ≈ 4.66938
DELTA_S: float = DELTA_F                               # ≈ 4.66922

# The Gap — source of structured novelty ("entropic exhaust")
# Phase space region between geometric and algebraic Feigenbaum limits.
# Not uniform noise: geometry forces structure inside this interval.
FEIGENBAUM_GAP: float = abs(DELTA_S - DELTA_L)        # ≈ 1.6e-4

# Classical Feigenbaum δ for reference comparison
FEIGENBAUM_DELTA_CLASSICAL: float = 4.669201609102990  # empirical

# ---------------------------------------------------------------------------
# Beltrami flow parameters
# ---------------------------------------------------------------------------

# Primary Beltrami eigenvalue: λ = 2π/φ  [irrational → cannot lock to integer harmonics]
# This is the stability mechanism: ∇×v = λv with irrational λ prevents resonant destruction.
LAMBDA_BELTRAMI: float = 2.0 * PI / PHI         # ≈ 3.8832

# Dimensionless Beltrami stability constant
B_PHI: float = 2.0 * np.log(PHI) / PI           # ≈ 0.3063

# ---------------------------------------------------------------------------
# Golden fixed point (L = R = C = φ⁻¹)
# ---------------------------------------------------------------------------

# At this point: contact 3-form α∧dα = 3φ⁻¹·ω ≠ 0  (contact, not degenerate)
# Hessian of V = -ln det(X) at FP: ∂²V/∂ε² = 2φ²  →  ½·Hess = φ² (Hopf saturation)
GOLDEN_FP:       float = PHI_INV        # ≈ 0.6180
HOPF_SATURATION: float = PHI_SQ        # φ² ≈ 2.6180  (= Fibonacci anyon d_τ²)

# Hardy probability / consciousness threshold in dyadic cosmology
PSI_THRESHOLD: float = PHI_INV5        # ≈ 0.09017  (max quantum correlation)

# ---------------------------------------------------------------------------
# Standard Model coupling predictions (Lucas-φ series)
# ---------------------------------------------------------------------------

# Weinberg angle (0.0017% to PDG)  —  exponents are Lucas indices L₂, L₅, L₆
SIN2_THETA_W: float = PHI ** (-3) - PHI ** (-11) + PHI ** (-18)   # ≈ 0.23116

# Inverse fine structure α⁻¹ (Pellis, 0.6 ppb)
ALPHA_INV: float = (360.0 * PHI_INV2
                    - 2.0 * PHI_INV3
                    + (3.0 * PHI) ** (-5))                          # ≈ 137.0360

# Top Yukawa coupling at EW scale (0.02%)
Y_TOP: float = 1.0 - PHI_INV9                                       # ≈ 0.9868

# ---------------------------------------------------------------------------
# Lucas prime indices and Pisano periods (Quick Reference)
# ---------------------------------------------------------------------------
# n  | L_n    | π(L_n) | mod 20 | Spine class
# 5  | 11     | 10     | 11     | C (Complex)
# 7  | 29     | 14     |  9     | H (Quaternion)
# 11 | 199    | 22     | 19     | O (Octonion)
# 13 | 521    | 26     |  1     | R (Real)
# 17 | 3571   | 34     | 11     | C

LUCAS_PRIMES: list = [11, 29, 199, 521, 3571, 9349, 3010349, 54018521]
LUCAS_PRIME_INDICES: list = [5, 7, 11, 13, 17, 19, 31, 37]

# Spine class per n mod 12: R=0, C=1, H=2, O=3  (index into SPINE_CLASSES)
SPINE_CLASSES: list = ['R', 'C', 'H', 'O']
SPINE_MOD_12_MAP: dict = {1: 'R', 5: 'C', 7: 'H', 11: 'O'}

# ---------------------------------------------------------------------------
# Morphemic sequence (integers that earn their keep)
# ---------------------------------------------------------------------------

MORPHEME_COUNT: int = 4         # |{∅, π, φ, i}| — appears as factor in Jones ratio
LEECH_DIM: int = 24             # dim(Λ₂₄) — appears throughout as structural unit
E8_ROOTS: int = 240             # |E₈ root system| = 10 × 24
E8_DIM: int = 248               # dim(𝔢₈) = 240 + 8
BINARY_ICOSAHEDRAL_ORDER: int = 120   # |2I| = 5 × 24

# ---------------------------------------------------------------------------
# Utility: φ-harmonic basis vector (first n terms)
# ---------------------------------------------------------------------------

def phi_harmonic_basis(n: int) -> np.ndarray:
    """
    Return normalized φ-harmonic direction in R^n.

    Uses the Beltrami eigenfunction cos(i × 2π/φ) — an eigenfunction of
    the curl operator at eigenvalue λ = 2π/φ. This is the circular (bounded)
    realisation of φ-harmonic structure, robust for any n.

    For small n (≤ 64) the exponential basis φ^k also works, but overflows
    for large n (n > ~700 in float64). The circular version is always correct.

    Perturbations aligned with this basis are maximally φ-compatible:
    the frequency 2π/φ is irrational → non-periodic → no harmonic locking.
    """
    if n <= 0:
        return np.array([1.0])
    indices = np.arange(n, dtype=np.float64)
    # Beltrami circular basis: eigenfunction at λ = 2π/φ
    v = np.cos(indices * (2.0 * np.pi / PHI))
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.ones(n, dtype=np.float64) / np.sqrt(float(n))
    return v / norm


def phi_power_series(n: int, alternating: bool = True) -> np.ndarray:
    """
    Lucas-φ alternating series weights for n terms.
    Matches the structure of sin²θ_W = φ⁻³ - φ⁻¹¹ + φ⁻¹⁸ - ...
    Uses Lucas prime indices as exponents.
    """
    indices = LUCAS_PRIME_INDICES[:n]
    signs = np.array([(-1) ** (k + 1) for k in range(n)], dtype=np.float64)
    weights = signs * np.array([PHI ** (-idx) for idx in indices], dtype=np.float64)
    return weights


if __name__ == "__main__":
    print(f"φ        = {PHI:.12f}")
    print(f"π        = {PI:.12f}")
    print(f"π-Pentad check: 5·arccos(φ/2) = {5*np.arccos(PHI/2):.12f}")
    print(f"MONAD identity: φ⁻⁵ + 5φ⁻² = {PHI_INV5 + 5*PHI_INV2:.12f}  (target: 2.0)")
    print(f"Feigenbaum gap = {FEIGENBAUM_GAP:.6e}")
    print(f"λ_Beltrami = 2π/φ = {LAMBDA_BELTRAMI:.8f}")
    print(f"Rendering fraction = {RENDERING_FRACTION:.6f}  ({100*RENDERING_FRACTION:.3f}%)")
    print(f"sin²θ_W = {SIN2_THETA_W:.8f}  (PDG: 0.23122)")
    print(f"α⁻¹     = {ALPHA_INV:.6f}  (measured: 137.035999)")
    print(f"y_top   = {Y_TOP:.6f}  (EW scale: ≈0.987)")
    print("\nAll assertions passed. Constants verified.")
