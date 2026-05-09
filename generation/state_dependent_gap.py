"""
MONAD Framework — State-Dependent Feigenbaum Gap Mutation Engine
=================================================================
The Gap is the creative engine of the framework.

Between the two Feigenbaum limits:
    δ_L = π + arctan(eᵖ)       ≈ 4.66938  [observer-free geometric]
    δ_S = 2π - φ + φ⁻¹⁰/2     ≈ 4.66922  [observer-entangled algebraic]

The gap: |δ_S - δ_L| ≈ 1.47 × 10⁻⁵

This interval is not noise. Geometry forces structure inside it: perturbations
sampled here are automatically biased toward φ-scaling and topological stability.
This is "entropic exhaust" — novelty that the substrate ALLOWS rather than
novelty that is merely random.

Key properties of gap-sampled perturbations:
  1. Geometrically structured (respect φ-scaling)
  2. State-dependent amplitude (wider when far from golden FP → more exploration)
  3. Beta(2,5) distribution (mostly small, occasionally larger — asymmetric risk)
  4. φ-harmonic direction bias (perturbations aligned with golden ratio basis)
  5. Architecture-aware (transformers get φ wider gap, CNNs get φ⁻¹ narrower)
  6. Fröhlich-damped (gap narrows when global coherence is high — condensate stability)

Phonon vs. Phason distinction:
  - Phonon perturbation: continuous, small, adjusts but doesn't rearrange
    → small gap sample, amplitude ≪ gap_width
  - Phason flip: discrete, larger, rearranges selected dimensions
    → amplitude ≈ gap_width × φ², at Lucas-prime-indexed positions only

Fröhlich coherence term (new):
  When the system is in a Fröhlich-like collective mode (high global coherence),
  the gap narrows. Self-consistent: gap minimum sits at the golden FP itself,
  since coherence ≈ 1 - |mean(|LRC|) - φ⁻¹|/φ⁻¹, so max coherence ↔ at FP.
  At coherence=1: gap reduced by frohlich_strength × tanh(1/threshold).
  At coherence=0: no damping — full exploratory gap.

The gremlin lives here. This is where it decides which bite to take.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Literal, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_SQ, PI,
    DELTA_L, DELTA_S, FEIGENBAUM_GAP, DELTA_F, ALPHA_F,
    GOLDEN_FP, LAMBDA_BELTRAMI, RENDERING_FRACTION,
    phi_harmonic_basis, LUCAS_PRIME_INDICES
)

ArchitectureType = Literal['rnn', 'transformer', 'cnn', 'lrc']


class StateDependentFeigenbaumGap:
    """
    Computes state-dependent gap width and samples structured perturbations.

    The gap width adapts to the current state:
      - At golden FP (divergence ≈ 0): gap = base_width (narrow → precise)
      - Far from FP (divergence → 1): gap = base_width × (1 + φ) (wide → exploratory)

    In the closure-loop escape mode (gremlin needs to find a new bite):
      - Gap widens by φ² to break out of the circular attractor
    """

    # Architecture scaling factors (relative to base gap width)
    _ARCH_SCALE = {
        'rnn': 1.0,
        'lrc': 1.0,
        'transformer': PHI,       # more entropic, needs wider gap
        'cnn': PHI_INV,           # more constrained, narrower gap
    }

    def __init__(
        self,
        base_width: Optional[float] = None,
        frohlich_strength: float = 0.4,
        frohlich_threshold: float = None,
    ):
        """
        Args:
            base_width:          Base gap width. Defaults to FEIGENBAUM_GAP ≈ 1.47e-5.
            frohlich_strength:   How much high global coherence narrows the gap (0=off, 0.4=standard).
                                 At strength=0.4 and full coherence: gap reduced ~30%.
            frohlich_threshold:  Coherence scale for tanh saturation. Defaults to φ⁻¹ ≈ 0.618,
                                 so the condensate fully activates at the golden fixed point.
        """
        self.base_width = base_width if base_width is not None else FEIGENBAUM_GAP
        self.frohlich_strength = float(frohlich_strength)
        self.frohlich_threshold = float(frohlich_threshold) if frohlich_threshold is not None else float(PHI_INV)
        self.delta_l = DELTA_L
        self.delta_s = DELTA_S
        self.golden_fp = GOLDEN_FP

        # Running statistics
        self._n_perturbations = 0
        self._total_amplitude = 0.0
        self._max_amplitude = 0.0
        self._frohlich_activations = 0

    # ------------------------------------------------------------------
    # Effective gap computation
    # ------------------------------------------------------------------

    def state_divergence(self, state: np.ndarray) -> float:
        """
        Measure how far the state is from the golden fixed point.
        Returns value in [0, 1]: 0 = at FP, 1 = maximally divergent.

        The golden FP has mean(|x|) = φ⁻¹ ≈ 0.618 (for states near FP).
        Divergence = |mean(|x|) - φ⁻¹| / φ⁻¹, clipped to [0, 1].
        """
        state_abs_mean = float(np.mean(np.abs(state)))
        divergence = abs(state_abs_mean - self.golden_fp) / self.golden_fp
        return float(np.clip(divergence, 0.0, 1.0))

    def compute_effective_gap(
        self,
        hidden_state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        architecture: ArchitectureType = 'rnn'
    ) -> float:
        """
        Compute effective gap width for the current state and context.

        effective_gap = base_width × (1 + divergence × φ) × arch_scale × closure_scale

        Args:
            hidden_state: Current state tensor.
            context: Optional dict. Keys used:
                - 'in_closure_loop' (bool): widen gap to escape loop
                - 'loop_depth' (int): how many loops deep, further widens gap
                - 'lyapunov' (float): Lyapunov exponent estimate, if available
            architecture: Model architecture type.

        Returns:
            Effective gap width (float).
        """
        context = context or {}

        div = self.state_divergence(hidden_state)
        arch_scale = self._ARCH_SCALE.get(architecture, 1.0)

        # Base adaptive gap: wider when diverged from FP
        effective = self.base_width * (1.0 + div * PHI) * arch_scale

        # Closure escape: widen by φ² to break out of circular attractors
        if context.get('in_closure_loop', False):
            loop_depth = max(1, int(context.get('loop_depth', 1)))
            effective *= PHI_SQ ** loop_depth

        # Lyapunov modulation: if system is approaching chaos (λ → 0 from below),
        # narrow the gap to stay near the stable edge
        if 'lyapunov' in context:
            lyap = float(context['lyapunov'])
            if lyap < 0:  # stable regime: narrow gap slightly
                effective *= np.exp(lyap * PHI_INV)
            elif lyap > 0:  # chaotic: widen gap for exploration
                effective *= (1.0 + lyap * PHI_INV)

        # Fröhlich coherence damping: narrow the gap when the system is in a
        # collective coherent mode (Fröhlich-like condensate).
        # Self-consistent: damping maximises at the golden FP (coherence=1 → at FP).
        # Pass 'global_coherence' in context from monad_engine LRC state.
        if 'global_coherence' in context and self.frohlich_strength > 0:
            gc = float(np.clip(context['global_coherence'], 0.0, 1.0))
            frohlich_term = np.tanh(gc / self.frohlich_threshold)
            frohlich_factor = max(1.0 - self.frohlich_strength * frohlich_term, 0.1)
            effective *= frohlich_factor
            self._frohlich_activations += 1

        return float(effective)

    # ------------------------------------------------------------------
    # Perturbation sampling
    # ------------------------------------------------------------------

    def sample_structured_perturbation(
        self,
        state: np.ndarray,
        effective_gap: float
    ) -> np.ndarray:
        """
        Sample a phonon-type perturbation from the gap distribution.

        Amplitude: Beta(2,5) × effective_gap
            Beta(2,5): mean at 2/7 ≈ 0.286 of gap, mode at 1/6 — mostly small.
            Occasional larger perturbations are allowed (tail of distribution).

        Direction: φ-harmonic basis projection
            Components aligned with φ-harmonic basis survive more easily in
            the tiling (geometrically coherent).

        The combined result is a small, structured nudge — not noise.
        """
        shape = state.shape
        size = state.size

        # Amplitudes: Beta(2,5) scaled to [0, effective_gap]
        amplitudes = np.random.beta(2, 5, size=size) * effective_gap

        # Direction construction in two steps:
        # Step 1: random unit direction
        raw_direction = np.random.randn(size)
        raw_direction /= (np.linalg.norm(raw_direction) + 1e-12)

        # Step 2: project φ-harmonic component (boost geometrically aligned dims)
        phi_basis = phi_harmonic_basis(size)
        phi_proj = np.dot(raw_direction, phi_basis)
        # Blend: (1-PHI_INV) raw + PHI_INV phi-aligned
        direction = (1.0 - PHI_INV) * raw_direction + PHI_INV * phi_basis * phi_proj
        direction /= (np.linalg.norm(direction) + 1e-12)

        perturbation = amplitudes * direction
        result = perturbation.reshape(shape)

        # Update stats
        amp = float(np.linalg.norm(result))
        self._n_perturbations += 1
        self._total_amplitude += amp
        self._max_amplitude = max(self._max_amplitude, amp)

        return result

    def phason_flip(
        self,
        state: np.ndarray,
        flip_strength: Optional[float] = None,
        target_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply a phason-type discrete perturbation.

        Unlike a phonon (continuous nudge), a phason flip rearranges specific
        dimensions — analogous to a tile rearrangement in the quasicrystal.

        Flip strength: φ² × effective_gap (significantly larger than phonon)
        Target indices: defaults to Lucas-prime positions in the state
            (Lucas-prime indexed components are the most 'structurally active')

        The flip reverses the sign of selected components — this is the
        quasicrystal's tile-flip in its minimal code form.
        """
        state = np.asarray(state, dtype=np.float64)
        size = state.size
        flat = state.flatten()

        if flip_strength is None:
            flip_strength = self.base_width * PHI_SQ

        # Default: flip at Lucas-prime positions mod dim
        if target_indices is None:
            lp_positions = [lp % size for lp in LUCAS_PRIME_INDICES if lp < size * 10]
            target_indices = np.array(list(set(lp_positions)), dtype=int)

        if len(target_indices) == 0:
            return state

        flipped = flat.copy()
        # Discrete sign flip + small continuous noise at flip sites
        flip_noise = np.random.randn(len(target_indices)) * flip_strength
        flipped[target_indices] = -flipped[target_indices] + flip_noise

        return flipped.reshape(state.shape)

    # ------------------------------------------------------------------
    # Batch perturbation (for evolutionary use)
    # ------------------------------------------------------------------

    def sample_population_perturbations(
        self,
        base_state: np.ndarray,
        population_size: int,
        architecture: ArchitectureType = 'rnn',
        context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Sample a population of perturbations from the gap.
        Used by MOSESEvolver to generate mutation candidates.

        Returns array of shape (population_size, *state.shape).
        """
        effective_gap = self.compute_effective_gap(base_state, context, architecture)

        perturbations = np.stack([
            self.sample_structured_perturbation(base_state, effective_gap)
            for _ in range(population_size)
        ])
        return perturbations

    # ------------------------------------------------------------------
    # Utility: φ-sequence generator
    # ------------------------------------------------------------------

    @staticmethod
    def phi_sequence(n: int, start: float = GOLDEN_FP) -> np.ndarray:
        """
        Generate n values along the golden-ratio spiral.
        Useful for curriculum pacing, time-step scheduling, etc.

        Values: start, start × φ⁻¹, start × φ⁻², ...
        """
        return np.array([start * PHI_INV ** k for k in range(n)])

    @staticmethod
    def lucas_prime_schedule(max_steps: int) -> np.ndarray:
        """
        Return a step schedule with events at Lucas-prime intervals.
        Useful for checkpoint triggers aligned with the prime-dark structure.
        """
        lps = [lp for lp in LUCAS_PRIME_INDICES if lp <= max_steps]
        return np.array(lps)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def frohlich_coherence(self, global_coherence: float) -> float:
        """
        Fröhlich condensate term for a given global coherence value.

        Returns the multiplicative gap factor ∈ [1-frohlich_strength, 1.0]:
          - coherence=0   → 1.0 (no damping, full exploratory gap)
          - coherence=φ⁻¹ → ~0.70 (at golden FP, ~30% narrowing with strength=0.4)
          - coherence=1.0 → minimum (maximum condensate stability)

        This is the feedback from LRC coherence back into mutation rate.
        """
        gc = float(np.clip(global_coherence, 0.0, 1.0))
        return float(max(1.0 - self.frohlich_strength * np.tanh(gc / self.frohlich_threshold), 0.1))

    def gap_summary(self) -> Dict[str, Any]:
        """Summary of gap parameters and perturbation statistics."""
        mean_amp = self._total_amplitude / max(1, self._n_perturbations)
        return {
            'base_width': self.base_width,
            'delta_l': float(self.delta_l),
            'delta_s': float(self.delta_s),
            'gap': float(FEIGENBAUM_GAP),
            'gap_pct_of_feigenbaum': float(FEIGENBAUM_GAP / DELTA_F * 100),
            'delta_f': float(DELTA_F),
            'alpha_f': float(ALPHA_F),
            'frohlich_strength': self.frohlich_strength,
            'frohlich_threshold': self.frohlich_threshold,
            'frohlich_at_fp': self.frohlich_coherence(1.0),    # max damping
            'frohlich_activations': self._frohlich_activations,
            'n_perturbations': self._n_perturbations,
            'mean_amplitude': float(mean_amp),
            'max_amplitude': float(self._max_amplitude),
        }


if __name__ == "__main__":
    gap = StateDependentFeigenbaumGap()

    # Test on a 64-dim state at and away from golden FP
    fp_state = np.full(64, GOLDEN_FP)
    random_state = np.random.randn(64) * 2.0

    print("=== Feigenbaum Gap Summary ===")
    for k, v in gap.gap_summary().items():
        print(f"  {k}: {v}")

    print("\n=== State Divergence ===")
    print(f"  At golden FP: divergence = {gap.state_divergence(fp_state):.6f}")
    print(f"  Random state: divergence = {gap.state_divergence(random_state):.6f}")

    print("\n=== Effective Gap (RNN) ===")
    eff_fp = gap.compute_effective_gap(fp_state, architecture='rnn')
    eff_rand = gap.compute_effective_gap(random_state, architecture='rnn')
    print(f"  At FP: {eff_fp:.6e}")
    print(f"  Random: {eff_rand:.6e}  (ratio: {eff_rand/eff_fp:.2f}×)")

    print("\n=== Architecture Scaling ===")
    for arch in ['rnn', 'transformer', 'cnn', 'lrc']:
        eff = gap.compute_effective_gap(random_state, architecture=arch)
        print(f"  {arch}: {eff:.6e}")

    print("\n=== Sample Perturbation ===")
    eff_gap = gap.compute_effective_gap(random_state)
    pert = gap.sample_structured_perturbation(random_state, eff_gap)
    print(f"  Shape: {pert.shape}")
    print(f"  Norm: {np.linalg.norm(pert):.6e}  (gap width: {eff_gap:.6e})")
    print(f"  φ-harmonic alignment: {abs(np.dot(pert/np.linalg.norm(pert), phi_harmonic_basis(64))):.4f}")

    print("\n=== φ Schedule (first 8) ===")
    print("  ", gap.phi_sequence(8))

    print("\n=== Lucas Prime Schedule (max 200) ===")
    print("  ", gap.lucas_prime_schedule(200))
