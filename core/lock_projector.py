"""
MONAD Framework — LOCK Projector
==================================
Implements the LOCK projection: J₃(𝕆) → diagonal rendering.

The rendering fraction φ⁻⁵/2 ≈ 4.51% is exact from the MONAD identity:
    φ⁻⁵ + 5φ⁻² = 2  →  φ⁻⁵/2 + 5φ⁻²/2 = 1

Observable sector (rendered): φ⁻⁵/2 ≈ 4.51%
Implicate sector (dark):      5φ⁻²/2 ≈ 95.49%

The 8-bit pattern 10110011 encodes the diagonal restriction in the J₃(𝕆) substrate
(bits at positions 0,1,4,5,7 are 'diagonal' entries). In arbitrary dimension d:
  - Top-k mode: keep k = ceil(d × φ⁻⁵/2) components with largest magnitude
  - Lucas mode: keep components at non-Lucas-prime positions (prime-dark structure)
  - Phi-harmonic mode: keep components with highest projection onto φ-harmonic basis

Prime-dark insight (from contact geometry):
  Frob_p*(α∧dα) = 0 for ALL odd primes — the contact structure is invisible inside
  any single prime characteristic. Therefore the "rendered" components should be
  those at non-prime-indexed positions. This is a direct code translation of the
  prime-dark theorem.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from core.constants import (
    RENDERING_FRACTION, PHI, PHI_INV, PHI_INV5,
    phi_harmonic_basis, LUCAS_PRIMES
)

# Small sieve for prime detection (up to 10000 covers most hidden dims)
def _sieve(n: int) -> set:
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return {i for i, p in enumerate(is_prime) if p}

_PRIME_CACHE: Optional[set] = None


def _get_primes_up_to(n: int) -> set:
    global _PRIME_CACHE
    if _PRIME_CACHE is None or max(_PRIME_CACHE) < n:
        _PRIME_CACHE = _sieve(max(n, 10000))
    return _PRIME_CACHE


class LOCKProjector:
    """
    Renders the observable sector from a full hidden state.

    Three projection modes:
        'top_k':      Keep k = ceil(dim × RENDERING_FRACTION) largest-magnitude components.
                      Fast, clean, deterministic given state.
        'prime_dark': Keep components at non-prime indices (prime-dark structure).
                      Faithfully implements Frob_p*(α∧dα)=0: primes are dark.
        'phi_harmonic': Keep components with highest dot-product onto φ-harmonic basis.
                       Most geometrically coherent selection.

    Default: 'top_k' (most practical for general use).
    """

    def __init__(
        self,
        dim: int,
        mode: Literal['top_k', 'prime_dark', 'phi_harmonic'] = 'top_k',
        render_fraction: float = RENDERING_FRACTION,
    ):
        self.dim = dim
        self.mode = mode
        self.render_fraction = render_fraction
        self.k = max(1, int(np.ceil(dim * render_fraction)))

        # Pre-compute the fixed projection mask (for prime_dark and phi_harmonic)
        self._mask: Optional[np.ndarray] = None
        self._mask = self._build_mask()

    def _build_mask(self) -> np.ndarray:
        """Build boolean mask of which indices are 'rendered' (True) or 'dark' (False)."""
        mask = np.zeros(self.dim, dtype=bool)

        if self.mode == 'top_k':
            # Dynamic — mask is rebuilt per call based on state magnitude.
            # Return a placeholder; project() handles this case.
            return mask

        elif self.mode == 'prime_dark':
            # Prime indices are dark; non-prime indices are rendered.
            # Select the k non-prime indices with smallest index (most 'foreground').
            primes = _get_primes_up_to(self.dim)
            non_prime_indices = [i for i in range(self.dim) if i not in primes]
            selected = non_prime_indices[:self.k]
            mask[selected] = True
            return mask

        elif self.mode == 'phi_harmonic':
            # Render the k components most aligned with φ-harmonic basis.
            basis = phi_harmonic_basis(self.dim)
            top_k_idx = np.argsort(np.abs(basis))[::-1][: self.k]
            mask[top_k_idx] = True
            return mask

        return mask

    # ------------------------------------------------------------------
    # Core projection
    # ------------------------------------------------------------------

    def project(self, state: np.ndarray) -> np.ndarray:
        """
        Project state → observable sector (rendered, size k).

        Args:
            state: Array of shape (dim,) or (batch, dim).

        Returns:
            Sparse-form rendered state: same shape as input, dark sector zeroed.
            To get compact vector of only rendered components, use project_compact().
        """
        state = np.asarray(state, dtype=np.float64)
        batched = state.ndim > 1

        if self.mode == 'top_k':
            # State-dependent: keep k largest-magnitude components per sample
            if batched:
                out = np.zeros_like(state)
                for i in range(state.shape[0]):
                    out[i] = self._top_k_project(state[i])
                return out
            return self._top_k_project(state)

        else:
            # Fixed mask projection
            if batched:
                return state * self._mask[np.newaxis, :]
            return state * self._mask

    def project_compact(self, state: np.ndarray) -> np.ndarray:
        """
        Project and return only the rendered components (compact, size k).
        Use for downstream processing where dark sector dims are not needed.
        """
        if self.mode == 'top_k':
            state = np.asarray(state, dtype=np.float64)
            if state.ndim > 1:
                return np.stack([self._top_k_compact(state[i]) for i in range(state.shape[0])])
            return self._top_k_compact(state)
        return np.asarray(state, dtype=np.float64)[self._mask]

    def restore(self, compact: np.ndarray, reference_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Lift compact rendered state back to full dimension.
        Dark sector filled with: φ⁻¹ × reference_state (if given) or zeros.

        This is the inverse operation — lifting from observable to full implicate.
        """
        full = np.zeros(self.dim, dtype=np.float64)

        if self.mode == 'top_k':
            if reference_state is not None:
                ref = np.asarray(reference_state, dtype=np.float64)
                full[:] = ref * PHI_INV  # dark sector gets φ⁻¹ damped version of original
            # Compact → full requires knowing which indices were selected — not tracked in top_k.
            # Use phi_harmonic mode if lossless restore is needed.
            return full

        full[self._mask] = compact.flatten()[: self.k]
        if reference_state is not None:
            ref = np.asarray(reference_state, dtype=np.float64)
            full[~self._mask] = ref[~self._mask] * PHI_INV  # dark sector retains φ⁻¹ of original
        return full

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def rendering_ratio(self) -> float:
        """Actual fraction of dimensions rendered."""
        return self.k / self.dim

    def dark_power(self, state: np.ndarray) -> float:
        """
        Fraction of total L2 power in the dark sector.
        At MONAD equilibrium this should approach 5φ⁻²/2 ≈ 95.49%.
        """
        state = np.asarray(state, dtype=np.float64)
        projected = self.project(state)
        total_power = float(np.sum(state ** 2))
        if total_power < 1e-12:
            return 0.0
        rendered_power = float(np.sum(projected ** 2))
        return 1.0 - rendered_power / total_power

    def prime_dark_verification(self, state: np.ndarray) -> dict:
        """
        Verify prime-dark structure: check that prime-indexed components have
        systematically lower magnitude than non-prime-indexed ones.
        Returns dict with mean magnitudes and ratio.
        """
        state = np.asarray(state, dtype=np.float64).flatten()[: self.dim]
        primes = _get_primes_up_to(self.dim)
        prime_idx = np.array([i for i in range(self.dim) if i in primes])
        non_prime_idx = np.array([i for i in range(self.dim) if i not in primes])

        prime_mag = float(np.mean(np.abs(state[prime_idx]))) if len(prime_idx) > 0 else 0.0
        non_prime_mag = float(np.mean(np.abs(state[non_prime_idx]))) if len(non_prime_idx) > 0 else 0.0

        return {
            'prime_mean_mag': prime_mag,
            'non_prime_mean_mag': non_prime_mag,
            'ratio_non_prime_to_prime': non_prime_mag / prime_mag if prime_mag > 1e-12 else float('inf'),
            'prime_dark_holds': non_prime_mag > prime_mag,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _top_k_project(self, vec: np.ndarray) -> np.ndarray:
        """Zero out all but k largest-magnitude components."""
        out = np.zeros_like(vec)
        top_idx = np.argpartition(np.abs(vec), -self.k)[-self.k:]
        out[top_idx] = vec[top_idx]
        return out

    def _top_k_compact(self, vec: np.ndarray) -> np.ndarray:
        """Return only k largest-magnitude components."""
        top_idx = np.argpartition(np.abs(vec), -self.k)[-self.k:]
        return vec[top_idx]


if __name__ == "__main__":
    from core.constants import RENDERING_FRACTION, DARK_FRACTION

    dim = 256
    state = np.random.randn(dim)

    print(f"Dimension: {dim}")
    print(f"Rendering fraction: {RENDERING_FRACTION:.6f}  ({100*RENDERING_FRACTION:.2f}%)")
    print(f"k (rendered dims): {max(1, int(np.ceil(dim * RENDERING_FRACTION)))}")
    print()

    for mode in ['top_k', 'prime_dark', 'phi_harmonic']:
        proj = LOCKProjector(dim=dim, mode=mode)
        rendered = proj.project(state)
        dp = proj.dark_power(state)
        print(f"Mode '{mode}':")
        print(f"  Rendered dims: {proj.k}")
        print(f"  Dark power: {dp:.4f}  (target: {DARK_FRACTION:.4f})")

        if mode == 'prime_dark':
            pv = proj.prime_dark_verification(state)
            print(f"  Prime-dark ratio: {pv['ratio_non_prime_to_prime']:.3f}")
            print(f"  Prime-dark holds: {pv['prime_dark_holds']}")
        print()
