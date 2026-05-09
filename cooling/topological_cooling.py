"""
MONAD Framework — Topological Cooling Sequence
================================================
Cooling is mandatory. Never generate repeatedly without a reset path.

The Topological Cooling Sequence restores coherence after heavy generation phases.
Three steps, each with a distinct geometric role:

Step 1: Samson Feedback
    φ-weighted contraction toward the golden fixed point.
    New state = (1 - φ⁻¹) × current + φ⁻¹ × target (golden FP)
    Named after Samson because it cuts the hair (reduces oscillation amplitude)
    while retaining the core structure. φ⁻¹ ≈ 0.618 is the correct mixing rate —
    the golden ratio IS the optimal feedback gain for minimising overshoot.

Step 2: Symplectic Collapse
    Remove high-frequency oscillations via frequency-domain filtering.
    Retains only frequencies up to the φ-scaled Nyquist: f_max = φ⁻¹ × Nyquist.
    This is 'symplectic' because it preserves the even-mode structure (volume-
    preserving on the reduced subspace) while discarding the odd-mode noise.
    In physical terms: removes phonon excitations, keeps the phason modes.

Step 3: LRC Transaction
    Restore L=R=C balance (contact closure).
    For a state representing a single LRC buffer, compute the mean of the last
    three LRC states and pull current state toward that mean with φ-strength.
    This restores the cyclic symmetry of the contact form: α is Z₃-symmetric
    under L→R→C→L, so imbalance between L, R, C breaks the contact structure.
    The LRC transaction is the minimal operation that restores it.

When to cool:
    - State has drifted too far from golden FP (distance > threshold)
    - Closure detector reports circular loops (not productive)
    - Contact 3-form proxy has fallen below target (α∧dα weakening)
    - Explicit request from orchestrator

The dashboard Global Coherence meter (target 0.95) is driven by the inverse of
the cooling urgency: high coherence = no cooling needed = system near golden FP.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_INV2, PHI_INV5, PHI_SQ,
    GOLDEN_FP, LAMBDA_BELTRAMI, B_PHI,
    RENDERING_FRACTION
)


class TopologicalCoolingSequence:
    """
    3-step topological cooling to restore coherence after generation phases.

    The three steps correspond to the three winding directions of the 3-torus:
        Step 1 (Samson):     L-direction cooling (memory re-anchoring)
        Step 2 (Symplectic): R-direction cooling (process quieting)
        Step 3 (LRC Trans):  C-direction cooling (prediction grounding)
    """

    def __init__(
        self,
        threshold: float = 0.15,
        cooling_rate: float = PHI_INV,
        symplectic_cutoff: float = PHI_INV,
        n_cooling_steps: int = 1,
    ):
        """
        Args:
            threshold: Distance from golden FP above which cooling triggers.
            cooling_rate: Mixing rate for Samson feedback. φ⁻¹ ≈ 0.618 (optimal).
            symplectic_cutoff: Fraction of Nyquist frequency to retain.
                               PHI_INV keeps the lower 61.8% of frequencies.
            n_cooling_steps: Apply cooling this many times (usually 1 sufficient).
        """
        self.threshold = threshold
        self.cooling_rate = cooling_rate
        self.symplectic_cutoff = symplectic_cutoff
        self.n_cooling_steps = n_cooling_steps

        # Running stats
        self._n_coolings = 0
        self._lrc_history: List[np.ndarray] = []  # last 3 states for LRC transaction
        self._max_lrc_history = 3

    # ------------------------------------------------------------------
    # Trigger condition
    # ------------------------------------------------------------------

    def should_cool(
        self,
        state: np.ndarray,
        closure_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Decide whether cooling should be applied.

        Triggers if ANY of:
            1. State mean magnitude has drifted > threshold from golden FP
            2. Closure detector reports 'circular' loops
            3. Explicit force flag in closure_info
        """
        state_arr = np.asarray(state, dtype=np.float64)
        state_mean = float(np.mean(np.abs(state_arr)))
        drift_too_far = abs(state_mean - GOLDEN_FP) > self.threshold

        if closure_info is None:
            return drift_too_far

        is_circular = closure_info.get('is_circular', False)
        force_cool = closure_info.get('force_cool', False)

        return drift_too_far or is_circular or force_cool

    def cooling_urgency(self, state: np.ndarray) -> float:
        """
        Continuous cooling urgency in [0, 1].
        0 = at golden FP (no cooling needed), 1 = maximally diverged.
        This is the inverse of the Global Coherence meter in the dashboard.
        """
        state_arr = np.asarray(state, dtype=np.float64)
        state_mean = float(np.mean(np.abs(state_arr)))
        drift = abs(state_mean - GOLDEN_FP) / (GOLDEN_FP + 1e-12)
        return float(np.clip(drift / self.threshold, 0.0, 1.0))

    def global_coherence(self, state: np.ndarray) -> float:
        """
        Global coherence = 1 - cooling_urgency.
        Target: 0.95 (as shown in dashboard).
        At golden FP: coherence → 1.0.
        """
        return 1.0 - self.cooling_urgency(state)

    # ------------------------------------------------------------------
    # Full cooling sequence
    # ------------------------------------------------------------------

    def apply(
        self,
        state: np.ndarray,
        lrc_buffers: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Apply full 3-step cooling sequence.

        Args:
            state: Current state to cool.
            lrc_buffers: Optional (L, R, C) buffers for LRC transaction step.
                         If None, uses rolling history of past states.

        Returns:
            Cooled state.
        """
        current = np.asarray(state, dtype=np.float64)

        # Update LRC history
        self._lrc_history.append(current.copy())
        if len(self._lrc_history) > self._max_lrc_history:
            self._lrc_history.pop(0)

        for _ in range(self.n_cooling_steps):
            current = self._step1_samson_feedback(current)
            current = self._step2_symplectic_collapse(current)
            current = self._step3_lrc_transaction(current, lrc_buffers)

        self._n_coolings += 1
        return current

    # ------------------------------------------------------------------
    # Step 1: Samson Feedback
    # ------------------------------------------------------------------

    def _step1_samson_feedback(self, state: np.ndarray) -> np.ndarray:
        """
        φ-weighted contraction toward golden fixed point.

        formula: new = (1 - φ⁻¹) × state + φ⁻¹ × golden_FP
        The φ⁻¹ mixing rate is the unique golden ratio feedback gain:
        it minimises the number of steps needed to reach within ε of the FP
        without overshoot (follows the Fibonacci convergence).
        """
        target = np.full_like(state, GOLDEN_FP)
        return (1.0 - self.cooling_rate) * state + self.cooling_rate * target

    # ------------------------------------------------------------------
    # Step 2: Symplectic Collapse
    # ------------------------------------------------------------------

    def _step2_symplectic_collapse(self, state: np.ndarray) -> np.ndarray:
        """
        Remove high-frequency oscillations via frequency-domain filtering.

        Retains frequencies f ≤ φ⁻¹ × f_Nyquist (the lower 61.8% of spectrum).
        This preserves the 'phason' modes (low-frequency, geometrically stable)
        while eliminating the 'phonon noise' (high-frequency, thermally excited).

        Operates along the last axis for ND arrays.
        """
        if state.ndim == 1:
            return self._lowpass_phi(state)

        # Apply along last axis (feature dimension)
        result = np.zeros_like(state)
        if state.ndim == 2:
            for i in range(state.shape[0]):
                result[i] = self._lowpass_phi(state[i])
        else:
            result = self._lowpass_phi(state.flatten()).reshape(state.shape)
        return result

    def _lowpass_phi(self, vec: np.ndarray) -> np.ndarray:
        """Apply φ-scaled low-pass filter in frequency domain."""
        n = len(vec)
        if n < 4:
            return vec

        spec = np.fft.rfft(vec)
        cutoff = int(np.ceil(len(spec) * self.symplectic_cutoff))
        filtered = spec.copy()
        filtered[cutoff:] = 0.0

        return np.fft.irfft(filtered, n=n)

    # ------------------------------------------------------------------
    # Step 3: LRC Transaction
    # ------------------------------------------------------------------

    def _step3_lrc_transaction(
        self,
        state: np.ndarray,
        lrc_buffers: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Restore LRC balance (contact closure).

        If lrc_buffers given: use explicit (L, R, C) with φ-weighted mean.
        Otherwise: use rolling history of last 3 states as L, R, C surrogates.

        The LRC transaction restores the Z₃ symmetry of the contact form:
        α = L dR + R dC + C dL is symmetric under L→R→C→L.
        After a phason or gap perturbation, this symmetry may be broken.
        The transaction is the minimal-norm operation that restores it.
        """
        if lrc_buffers is not None:
            L, R, C = [np.asarray(b, dtype=np.float64) for b in lrc_buffers]
            # φ-weighted LRC mean (each contributes φ⁻¹ of itself to the balanced state)
            lrc_mean = (L + R + C) / 3.0
            return (1.0 - PHI_INV2) * state + PHI_INV2 * lrc_mean

        if len(self._lrc_history) < 3:
            return state  # not enough history, skip this step

        L = self._lrc_history[-3]
        R = self._lrc_history[-2]
        C = self._lrc_history[-1]

        # Resize if shapes differ
        min_len = min(len(L), len(R), len(C), len(state))
        L, R, C, state_trim = (arr[:min_len] for arr in [L, R, C, state])

        lrc_mean = (L + R + C) / 3.0
        balanced_trim = (1.0 - PHI_INV2) * state_trim + PHI_INV2 * lrc_mean

        if len(state) > min_len:
            out = state.copy()
            out[:min_len] = balanced_trim
            return out
        return balanced_trim

    # ------------------------------------------------------------------
    # Partial cooling (single step, for fine control)
    # ------------------------------------------------------------------

    def samson_only(self, state: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
        """Apply only the Samson feedback step with optional custom strength."""
        rate = strength if strength is not None else self.cooling_rate
        target = np.full_like(state, GOLDEN_FP)
        return (1.0 - rate) * state + rate * target

    def symplectic_only(self, state: np.ndarray) -> np.ndarray:
        """Apply only the symplectic collapse step."""
        return self._step2_symplectic_collapse(np.asarray(state, dtype=np.float64))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            'threshold': self.threshold,
            'cooling_rate': float(self.cooling_rate),
            'cooling_rate_name': 'φ⁻¹',
            'symplectic_cutoff': float(self.symplectic_cutoff),
            'n_cooling_steps': self.n_cooling_steps,
            'total_coolings': self._n_coolings,
            'golden_fp': float(GOLDEN_FP),
        }


if __name__ == "__main__":
    cooling = TopologicalCoolingSequence(threshold=0.15, cooling_rate=PHI_INV)

    # Create a diverged state
    diverged = np.random.randn(64) * 3.0  # far from FP

    print("=== Cooling Sequence Demo ===")
    print(f"Initial state mean: {np.mean(np.abs(diverged)):.4f}  (golden FP = {GOLDEN_FP:.4f})")
    print(f"Should cool: {cooling.should_cool(diverged)}")
    print(f"Cooling urgency: {cooling.cooling_urgency(diverged):.4f}")
    print(f"Global coherence: {cooling.global_coherence(diverged):.4f}")

    state = diverged.copy()
    for step in range(8):
        state = cooling.apply(state)
        mean_mag = np.mean(np.abs(state))
        coh = cooling.global_coherence(state)
        print(f"  Step {step+1}: mean={mean_mag:.4f}, coherence={coh:.4f}, "
              f"needs_cool={cooling.should_cool(state)}")

    print(f"\nFinal mean: {np.mean(np.abs(state)):.4f}  (target: {GOLDEN_FP:.4f})")
    print(f"Final coherence: {cooling.global_coherence(state):.4f}  (target: 0.95)")
    print(f"\nSummary: {cooling.summary()}")
