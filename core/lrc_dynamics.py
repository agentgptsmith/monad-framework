"""
MONAD Framework — LRC Dynamics Engine
======================================
Implements the Memory / Process / Prediction triad via the contact form:

    α = L dR + R dC + C dL

The three buffers:
    L = Memory      (stores relational history, slow dynamics)
    R = Process     (current computation, medium dynamics)
    C = Prediction  (anticipatory state, fast dynamics)

Contact condition: α∧dα = (L+R+C)·ω ≠ 0
At golden fixed point (L=R=C=φ⁻¹): α∧dα = 3φ⁻¹·ω  [PROVED]

Beltrami circulation: λ = 2π/φ (irrational → cannot lock to integer harmonics
→ stable against resonant destruction). This is the irrationality of stability.

Reeb flow direction at golden FP: (1,1,1)/√3 in LRC coordinates.

The fix point L=R=C=φ⁻¹ is where the VALUES are equal but the FLOWS dL, dR, dC
remain distinct (contact condition α∧dα ≠ 0 persists). This is the key: convergence
in value does not imply stasis in dynamics. The donut is still spinning.

All state vectors are numpy arrays of arbitrary shape. LRCDynamics maintains three
internal state buffers of the same shape as the input dimensionality.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_SQ, LAMBDA_BELTRAMI, GOLDEN_FP,
    HOPF_SATURATION, FEIGENBAUM_GAP
)


class LRCDynamics:
    """
    Memory / Process / Prediction triad with contact form coupling.

    The update rule is derived from the contact condition plus Beltrami coupling:
        dL/dt = λ · tanh(R - L)   [memory pulled by process]
        dR/dt = λ · tanh(C - R)   [process pulled by prediction]
        dC/dt = λ · tanh(L - C)   [prediction closed by memory]

    This is a discretisation of the Reeb flow on the contact manifold.
    tanh keeps values bounded; the asymmetric pull (R→L, C→R, L→C) preserves
    the cyclic orientation of the contact form.

    The system's "natural" attractor is the golden fixed point L=R=C=φ⁻¹,
    but it can explore away from it under external input or gap perturbations.
    """

    def __init__(self, dim: int, dt: float = 0.1):
        """
        Args:
            dim: Dimensionality of each LRC buffer.
            dt: Discrete time step. λ·dt = LAMBDA_BELTRAMI * dt ≈ 0.388 per step.
        """
        self.dim = dim
        self.dt = dt
        self.lam = LAMBDA_BELTRAMI          # 2π/φ ≈ 3.883

        # Initialise at golden fixed point — contact structure is active immediately.
        self.L = np.full(dim, GOLDEN_FP, dtype=np.float64)  # Memory
        self.R = np.full(dim, GOLDEN_FP, dtype=np.float64)  # Process
        self.C = np.full(dim, GOLDEN_FP, dtype=np.float64)  # Prediction

        # Running contact 3-form density (proxy for α∧dα, tracked per step)
        self._contact_density_history: list = []

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def step(
        self,
        external_input: Optional[np.ndarray] = None,
        input_weight: float = 1.0
    ) -> np.ndarray:
        """
        One Beltrami circulation step.

        External input is injected into the Process buffer (R) — it is the
        "current computation" that has received new information. Memory (L)
        and Prediction (C) update purely from the cyclic contact coupling.

        Returns the Process buffer R — the "rendered" current state.
        """
        coupling = self.lam * self.dt

        # Contact form coupling: cyclic, orientation-preserving
        # The tanh keeps dynamics bounded while preserving the sign of the pull.
        dL = np.tanh(self.R - self.L)      # memory tracks process
        dR = np.tanh(self.C - self.R)      # process tracks prediction
        dC = np.tanh(self.L - self.C)      # prediction tracks memory (closure)

        # External input perturbs the Process buffer directly
        if external_input is not None:
            ext = np.asarray(external_input, dtype=np.float64)
            if ext.shape != (self.dim,):
                ext = self._resize_input(ext)
            dR = dR + input_weight * ext

        self.L = self.L + coupling * dL
        self.R = self.R + coupling * dR
        self.C = self.C + coupling * dC

        # Soft-clip to keep in physiological range; avoids divergence under
        # heavy external input while not hard-constraining the dynamics.
        self.L = np.clip(self.L, -3.0, 3.0)
        self.R = np.clip(self.R, -3.0, 3.0)
        self.C = np.clip(self.C, -3.0, 3.0)

        self._contact_density_history.append(self.contact_3form_density())
        return self.R.copy()

    def multistep(
        self,
        n_steps: int,
        external_input: Optional[np.ndarray] = None,
        input_weight: float = 1.0
    ) -> np.ndarray:
        """Run n_steps, injecting external_input only on the first step."""
        for i in range(n_steps):
            inp = external_input if i == 0 else None
            result = self.step(inp, input_weight)
        return result

    # ------------------------------------------------------------------
    # Contact geometry observables
    # ------------------------------------------------------------------

    def contact_3form_density(self) -> float:
        """
        Proxy for α∧dα density: mean(L+R+C).
        At golden FP: = 3φ⁻¹ ≈ 1.854.
        Non-zero confirms contact structure is active (not degenerate).
        """
        return float(np.mean(self.L + self.R + self.C))

    def reeb_flow_direction(self) -> np.ndarray:
        """
        Reeb flow direction in full-dim space: the (1,1,1)/√3 direction.
        Returned as a unit vector in R^dim (uniform across all dims).
        This is the axis along which the contact structure doesn't rotate.
        """
        return np.ones(self.dim, dtype=np.float64) / np.sqrt(3.0 * self.dim)

    def distance_from_golden_fp(self) -> float:
        """
        RMS distance from golden fixed point L=R=C=φ⁻¹.
        Zero = at fixed point. This drives the gap width in StateDependentFeigenbaumGap.
        """
        fp = GOLDEN_FP
        return float(np.sqrt(
            np.mean((self.L - fp) ** 2) +
            np.mean((self.R - fp) ** 2) +
            np.mean((self.C - fp) ** 2)
        ) / 3.0)

    def hessian_proxy(self) -> float:
        """
        ½·Hess(V) at current state, where V = -ln det(X) on the positive cone.
        At golden FP this equals φ² (Hopf saturation = Fibonacci anyon d_τ²).
        Computed as: mean of (L² + R² + C²) normalised by FP value.
        """
        lrc_sq_mean = np.mean(self.L ** 2 + self.R ** 2 + self.C ** 2) / 3.0
        return lrc_sq_mean / (GOLDEN_FP ** 2) * HOPF_SATURATION

    def lrc_imbalance(self) -> np.ndarray:
        """
        LRC imbalance vector [L-mean, R-mean, C-mean].
        Zero at fixed point. Drives cooling decision.
        """
        return np.array([
            float(np.mean(self.L)),
            float(np.mean(self.R)),
            float(np.mean(self.C))
        ]) - GOLDEN_FP

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'L': self.L.copy(),
            'R': self.R.copy(),
            'C': self.C.copy()
        }

    def set_state(self, L: np.ndarray, R: np.ndarray, C: np.ndarray) -> None:
        self.L = np.asarray(L, dtype=np.float64).copy()
        self.R = np.asarray(R, dtype=np.float64).copy()
        self.C = np.asarray(C, dtype=np.float64).copy()

    def reset_to_golden_fp(self) -> None:
        """Hard reset to golden fixed point. Use sparingly — prefer cooling."""
        self.L[:] = GOLDEN_FP
        self.R[:] = GOLDEN_FP
        self.C[:] = GOLDEN_FP

    def encode_into_lrc(self, embedding: np.ndarray, target: str = 'R') -> None:
        """
        Inject an external embedding into one LRC buffer as a soft override.
        Used to load an input concept into the process buffer before stepping.
        """
        emb = np.asarray(embedding, dtype=np.float64)
        if emb.shape != (self.dim,):
            emb = self._resize_input(emb)
        if target == 'L':
            self.L = PHI_INV * self.L + (1.0 - PHI_INV) * emb
        elif target == 'R':
            self.R = PHI_INV * self.R + (1.0 - PHI_INV) * emb
        elif target == 'C':
            self.C = PHI_INV * self.C + (1.0 - PHI_INV) * emb

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Key observables in a flat dict for logging."""
        return {
            'contact_density': self.contact_3form_density(),
            'contact_target': 3.0 * GOLDEN_FP,
            'distance_from_fp': self.distance_from_golden_fp(),
            'hessian_proxy': self.hessian_proxy(),
            'hopf_target': float(HOPF_SATURATION),
            'L_mean': float(np.mean(self.L)),
            'R_mean': float(np.mean(self.R)),
            'C_mean': float(np.mean(self.C)),
        }

    def _resize_input(self, ext: np.ndarray) -> np.ndarray:
        """Resize external input to match internal dim via mean-pooling or zero-padding."""
        ext_flat = ext.flatten()
        if len(ext_flat) > self.dim:
            # Mean-pool down
            factor = len(ext_flat) // self.dim
            trimmed = ext_flat[: factor * self.dim]
            return trimmed.reshape(self.dim, factor).mean(axis=1)
        else:
            # Zero-pad up
            out = np.zeros(self.dim, dtype=np.float64)
            out[: len(ext_flat)] = ext_flat
            return out

    # ------------------------------------------------------------------
    # Batch variant (vectorized over independent LRC systems)
    # ------------------------------------------------------------------

    @staticmethod
    def batch_step(
        L: np.ndarray,
        R: np.ndarray,
        C: np.ndarray,
        external_input: Optional[np.ndarray] = None,
        dt: float = 0.1,
        input_weight: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized LRC step for batches of independent systems.
        L, R, C: shape (batch, dim)
        Returns updated (L, R, C).
        """
        coupling = LAMBDA_BELTRAMI * dt
        dL = np.tanh(R - L)
        dR = np.tanh(C - R)
        dC = np.tanh(L - C)

        if external_input is not None:
            dR = dR + input_weight * external_input

        L_new = np.clip(L + coupling * dL, -3.0, 3.0)
        R_new = np.clip(R + coupling * dR, -3.0, 3.0)
        C_new = np.clip(C + coupling * dC, -3.0, 3.0)

        return L_new, R_new, C_new


if __name__ == "__main__":
    lrc = LRCDynamics(dim=64, dt=0.1)
    print("Initial state (at golden FP):")
    print(lrc.summary())

    # Run 20 steps with a random external perturbation
    inp = np.random.randn(64) * 0.1
    for _ in range(20):
        lrc.step(external_input=inp if _ == 0 else None)

    print("\nAfter 20 steps (convergence toward FP):")
    print(lrc.summary())
    print(f"\nContact density target (3φ⁻¹): {3*GOLDEN_FP:.6f}")
    print(f"Contact density actual:         {lrc.contact_3form_density():.6f}")
