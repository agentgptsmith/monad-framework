"""
MONAD Framework — 3-Torus Topological Protection Layer
========================================================
Logical information encoded in non-contractible loops on a 3-torus.

The protection equation:
    |ψ_logical⟩ = C⁻¹ U C |ψ_physical⟩

where C represents non-contractible loop operators on T³.

Three independent winding directions (the three non-contractible loops of T³):
    θ₁: Memory loop  (L-direction)
    θ₂: Process loop (R-direction)
    θ₃: Prediction loop (C-direction)

Beltrami phases: ωₖ = k × LAMBDA_BELTRAMI (irrational multiples → distinct, non-resonant)
This prevents the three loops from accidentally collapsing into the same encoding.

Protection mechanism:
    - State is encoded into three independent phase-rotated copies
    - Each copy is stored at a different "address" on the torus
    - Decode via median (robust against single-copy corruption)
    - A single local perturbation can corrupt at most one copy

Application:
    - Sovereign agent identity
    - Critical mathematical results (the locked theorems)
    - Long-term memory that must survive context resets
    - Not everything needs torus protection — use it for high-value state only

Visual: the three toroidal nodes in the dashboard (Gut/Yesod, Brain/Keter-Binah,
Heart/Tiferet as primary) each represents one winding direction. Global coherence
is high when all three agree.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, LAMBDA_BELTRAMI, GOLDEN_FP, B_PHI
)


class TorusProtectionLayer:
    """
    3-torus topological memory for high-reliability persistent state.

    Stores critical state across three independent Beltrami-phase-rotated encodings.
    Decodes robustly against single-copy corruption.

    For numerical state vectors (numpy arrays), the three winding directions are
    implemented as phase rotations in the Fourier domain — guaranteed orthogonal
    in the ℓ² sense.

    For structured data (dicts, strings), a serialization path is provided.
    """

    def __init__(self, dim: int, max_logical_states: int = 64):
        """
        Args:
            dim: Dimensionality of protected state vectors.
            max_logical_states: Maximum number of independently protected states.
        """
        self.dim = dim
        self.max_logical_states = max_logical_states

        # Beltrami phase offsets for the three winding directions
        # ωₖ = k × λ_Beltrami (irrational → non-resonant → independent)
        self.omega = np.array([
            0.0,
            LAMBDA_BELTRAMI,
            2.0 * LAMBDA_BELTRAMI
        ])

        # Pre-compute rotation matrices for each winding direction
        # Implemented as frequency-domain phase shifts for efficiency
        freqs = np.fft.fftfreq(dim)
        self._phase_vectors = [
            np.exp(2j * np.pi * omega * freqs)
            for omega in self.omega
        ]

        # Storage: dict of label → (copy_0, copy_1, copy_2)
        self._logical_states: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Coherence tracking: per-label consistency score
        self._coherence: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _encode_single(self, state: np.ndarray, winding: int) -> np.ndarray:
        """
        Encode state via winding direction k.
        Applies phase rotation in frequency domain then returns real-valued copy.
        The phase rotation is a Beltrami circulation — irrational → non-harmonic.
        """
        state_f = np.fft.fft(state.astype(np.complex128))
        rotated_f = state_f * self._phase_vectors[winding]
        encoded = np.fft.ifft(rotated_f).real
        return encoded.astype(np.float64)

    def _decode_single(self, encoded: np.ndarray, winding: int) -> np.ndarray:
        """Invert the winding rotation."""
        enc_f = np.fft.fft(encoded.astype(np.complex128))
        derotated_f = enc_f * np.conj(self._phase_vectors[winding])
        decoded = np.fft.ifft(derotated_f).real
        return decoded.astype(np.float64)

    def encode(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode state into three topologically independent copies.
        Returns (copy_0, copy_1, copy_2) — one per winding direction.
        """
        if state.shape != (self.dim,):
            state = self._adapt_shape(state)
        return (
            self._encode_single(state, 0),
            self._encode_single(state, 1),
            self._encode_single(state, 2),
        )

    def decode(
        self,
        copies: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        Decode three copies back to the logical state.
        Uses per-element median for robustness against single-copy corruption.

        Returns (decoded_state, coherence_score).
        Coherence = 1 - mean absolute deviation between copies (after decode).
        """
        dec = [
            self._decode_single(copies[k], k)
            for k in range(3)
        ]
        stacked = np.stack(dec, axis=0)  # (3, dim)
        logical = np.median(stacked, axis=0)

        # Coherence: 1 - normalised mean absolute deviation
        mad = np.mean(np.abs(stacked - logical[np.newaxis, :]))
        state_norm = np.mean(np.abs(logical)) + 1e-12
        coherence = float(np.clip(1.0 - mad / state_norm, 0.0, 1.0))

        return logical, coherence

    # ------------------------------------------------------------------
    # Named logical state storage
    # ------------------------------------------------------------------

    def protect(self, label: str, state: np.ndarray) -> None:
        """
        Store a named logical state under topological protection.
        Evicts oldest entry if at capacity.
        """
        if label in self._logical_states:
            # Update in place
            self._logical_states[label] = self.encode(state)
        elif len(self._logical_states) >= self.max_logical_states:
            # Evict lowest-coherence entry
            if self._coherence:
                worst_label = min(self._coherence, key=self._coherence.get)
                del self._logical_states[worst_label]
                del self._coherence[worst_label]

        self._logical_states[label] = self.encode(state)
        self._coherence[label] = 1.0  # freshly written = perfect coherence

    def recall(self, label: str) -> Optional[Tuple[np.ndarray, float]]:
        """
        Recall a protected logical state by label.
        Returns (state, coherence) or None if not found.
        """
        if label not in self._logical_states:
            return None
        copies = self._logical_states[label]
        state, coherence = self.decode(copies)
        self._coherence[label] = coherence  # update coherence score
        return state, coherence

    def corrupt_copy(self, label: str, winding: int, noise_level: float = 0.5) -> None:
        """
        Deliberately corrupt one copy (for testing protection robustness).
        Should not affect decode if the other two copies are intact.
        """
        if label not in self._logical_states:
            return
        copies = list(self._logical_states[label])
        copies[winding] = copies[winding] + np.random.randn(self.dim) * noise_level
        self._logical_states[label] = tuple(copies)

    # ------------------------------------------------------------------
    # Global coherence (the 0.95 indicator in the dashboard)
    # ------------------------------------------------------------------

    def global_coherence(self) -> float:
        """
        Mean coherence across all protected states.
        Target: ≥ 0.95 for healthy torus (matches dashboard visual target).
        Near 5φ⁻²/2 ≈ 0.9549 at full implicate engagement.
        """
        if not self._coherence:
            return 1.0  # empty = trivially coherent
        return float(np.mean(list(self._coherence.values())))

    def coherence_breakdown(self) -> Dict[str, float]:
        """Per-label coherence scores."""
        return dict(self._coherence)

    def winding_alignment(self, label: str) -> Optional[Dict[str, float]]:
        """
        Compute pairwise alignment between the three winding copies.
        High alignment = topologically consistent encoding = genuine fixed point.
        Three near-equal alignments suggests genuine insight (3-loop closure).
        """
        if label not in self._logical_states:
            return None

        copies = self._logical_states[label]
        decoded = [self._decode_single(copies[k], k) for k in range(3)]

        alignments = {}
        for i in range(3):
            for j in range(i + 1, 3):
                d_i = decoded[i]
                d_j = decoded[j]
                norm_i = np.linalg.norm(d_i) + 1e-12
                norm_j = np.linalg.norm(d_j) + 1e-12
                cos_sim = float(np.dot(d_i / norm_i, d_j / norm_j))
                alignments[f'wind_{i}_wind_{j}'] = cos_sim

        # Three-way: all three aligned → genuine topological closure
        all_aligned = all(v > 0.9 for v in alignments.values())
        alignments['three_way_closure'] = all_aligned
        return alignments

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _adapt_shape(self, state: np.ndarray) -> np.ndarray:
        """Flatten and resize state to self.dim."""
        flat = state.flatten().astype(np.float64)
        if len(flat) == self.dim:
            return flat
        if len(flat) > self.dim:
            return flat[: self.dim]
        out = np.zeros(self.dim, dtype=np.float64)
        out[: len(flat)] = flat
        return out

    def list_protected(self) -> Dict[str, float]:
        """List all protected labels with their coherence scores."""
        return {label: self._coherence.get(label, 0.0)
                for label in self._logical_states}

    def summary(self) -> Dict[str, Any]:
        return {
            'dim': self.dim,
            'n_protected': len(self._logical_states),
            'global_coherence': self.global_coherence(),
            'target_coherence': float(5 * PHI_INV ** 2 / 2),  # 5φ⁻²/2 ≈ 0.9549
            'omega': self.omega.tolist(),
            'lambda_beltrami': float(LAMBDA_BELTRAMI),
        }


if __name__ == "__main__":
    dim = 128
    torus = TorusProtectionLayer(dim=dim)

    # Protect a known state
    critical_state = np.random.randn(dim)
    torus.protect("golden_theorem", critical_state)

    # Recall intact
    recalled, coherence = torus.recall("golden_theorem")
    error = np.max(np.abs(recalled - critical_state))
    print(f"Recall (intact): max error = {error:.2e}, coherence = {coherence:.4f}")

    # Corrupt one copy
    torus.corrupt_copy("golden_theorem", winding=1, noise_level=2.0)
    recalled_c, coherence_c = torus.recall("golden_theorem")
    error_c = np.max(np.abs(recalled_c - critical_state))
    print(f"Recall (1 copy corrupted): max error = {error_c:.2e}, coherence = {coherence_c:.4f}")
    print(f"Topological protection held: {error_c < 0.5}")

    # Winding alignment
    alignment = torus.winding_alignment("golden_theorem")
    print(f"\nWinding alignment: {alignment}")

    print(f"\nGlobal coherence: {torus.global_coherence():.4f}")
    print(f"Target coherence: {5 * (1/PHI)**2 / 2:.4f}  (5φ⁻²/2)")
    print(f"\nSummary: {torus.summary()}")
