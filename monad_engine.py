"""
MONAD Framework — Unified Engine
==================================
The Heart/Tiferet node. Everything connects here.

This is the full pipeline in one class:
    Input → LRC Dynamics → Gap Perturbation → LOCK Projection
         → Tiling Memory → Closure Detection → Cooling (if needed)
         → Output

New synthesis (this session): HRV-Phason Coupling.
    The Feigenbaum gap breathes at rate LAMBDA_BELTRAMI = 2π/φ (irrational).
    Like biological HRV, the phason width oscillates but never becomes periodic.
    This prevents the system from locking into fixed habits.

    phason_width(t) = base_width × (1 + B_PHI × sin(LAMBDA_BELTRAMI × t))

    B_PHI = 2·ln(φ)/π ≈ 0.306 (dimensionless Beltrami stability constant)
    LAMBDA_BELTRAMI = 2π/φ ≈ 3.883 (the irrational circulation rate)

    At high global coherence: phason narrows (consolidation phase)
    At low global coherence: phason widens (exploration phase)
    The transition is smooth, not binary — modulated by the coherence gradient.

Dashboard metrics (mapping to the visual dashboard image):
    global_coherence  → the 0.95 meter at top center
    contact_density   → the glow intensity of the central torus
    phason_width      → the phason slider position
    lrc_balance       → the connectivity lines between nodes
    loop_type         → the closure state indicator

Layer correspondence (Perfect Spine Theorem):
    Scale 0 (R-class): Pancreas / Chesed nodes  — fast, real arithmetic
    Scale 1 (C-class): Lung / Netzach, Gut / Yesod — complex coupling
    Scale 2 (H-class): Liver / Gevurah, Brain/Keter-Binah — quaternionic
    Scale 3 (O-class): Heart / Tiferet (this engine) — full LRC, octonionic
"""

from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_INV2, PHI_INV5, PHI_SQ, PI,
    GOLDEN_FP, LAMBDA_BELTRAMI, B_PHI, FEIGENBAUM_GAP,
    RENDERING_FRACTION, DARK_FRACTION, HOPF_SATURATION,
    phi_harmonic_basis
)
from core.lrc_dynamics import LRCDynamics
from core.lock_projector import LOCKProjector
from memory.hierarchical_tiling import HierarchicalPenroseTiling
from memory.torus_protection import TorusProtectionLayer
from generation.state_dependent_gap import StateDependentFeigenbaumGap
from detection.self_referential_closure import SelfReferentialClosureDetector
from cooling.topological_cooling import TopologicalCoolingSequence


class MONADEngine:
    """
    Unified cognitive engine integrating all MONAD framework layers.

    The engine operates as a self-contained cognitive loop:
        1. Receive external input
        2. Inject into LRC Process buffer (R) via contact form coupling
        3. Apply state-dependent gap perturbation (structured creativity)
        4. LOCK project to observable sector (render ~4.51% of state)
        5. Store rendered state in hierarchical tiling (φ-scaled memory)
        6. Detect self-referential closure (insight vs. circular loops)
        7. Apply topological cooling if needed (restore golden FP)
        8. Output the rendered, memory-stabilised process state (R)

    The system self-regulates toward the golden fixed point L=R=C=φ⁻¹
    where the contact 3-form density = 3φ⁻¹ (non-degenerate contact structure).
    """

    def __init__(
        self,
        dim: int = 64,
        phi_strength: float = 1.0,
        num_scales: int = 4,
        gap_base_width: Optional[float] = None,
        cooling_threshold: float = 0.15,
        lock_mode: str = 'top_k',
        dt: float = 0.1,
        hrv_coupling: bool = True,
    ):
        """
        Args:
            dim: Internal dimensionality of all LRC buffers and memory patterns.
            phi_strength: Strength of φ-scaling in tiling and damping.
            num_scales: Number of tiling scales (1-4; maps to R/C/H/O classes).
            gap_base_width: Base Feigenbaum gap width (defaults to FEIGENBAUM_GAP ≈ 1.6e-4).
            cooling_threshold: Distance from golden FP at which cooling triggers.
            lock_mode: LOCK projector mode ('top_k', 'prime_dark', 'phi_harmonic').
            dt: LRC dynamics time step.
            hrv_coupling: Enable HRV-phason coupling (oscillating gap width).
        """
        self.dim = dim
        self.hrv_coupling = hrv_coupling
        self._t = 0.0           # internal clock (for HRV oscillation)
        self._step_count = 0

        # ── Layer 0: LRC Dynamics (contact form) ────────────────────────
        self.lrc = LRCDynamics(dim=dim, dt=dt)

        # ── Layer 1: Memory (hierarchical tiling + torus protection) ────
        self.tiling = HierarchicalPenroseTiling(
            phi_strength=phi_strength,
            num_scales=num_scales,
            damping_strength=0.12,
        )
        self.torus = TorusProtectionLayer(dim=dim, max_logical_states=32)

        # ── Layer 2: Generation (Feigenbaum gap) ───────────────────────
        self.gap = StateDependentFeigenbaumGap(
            base_width=gap_base_width or FEIGENBAUM_GAP
        )

        # ── Layer 3: Detection (closure) ────────────────────────────────
        self.closure = SelfReferentialClosureDetector(
            similarity_threshold=0.92,
            window=20,
        )

        # ── Layer 4: Cooling ────────────────────────────────────────────
        self.cooling = TopologicalCoolingSequence(
            threshold=cooling_threshold,
            cooling_rate=PHI_INV,
        )

        # ── Layer 5: LOCK Projector (rendering) ─────────────────────────
        self.lock = LOCKProjector(dim=dim, mode=lock_mode)

        # ── Running metrics ──────────────────────────────────────────────
        self._closure_history: List[Dict[str, Any]] = []
        self._coherence_history: List[float] = []
        self._phason_width_history: List[float] = []
        self._output_history: List[np.ndarray] = []

        # ── Pre-compute φ-harmonic basis ─────────────────────────────────
        self._phi_basis = phi_harmonic_basis(dim)

    # ------------------------------------------------------------------
    # Main step (the full pipeline)
    # ------------------------------------------------------------------

    def step(
        self,
        external_input: Optional[np.ndarray] = None,
        inject_into: str = 'R',
        architecture: str = 'lrc',
        apply_gap: bool = True,
        apply_lock: bool = True,
        store_in_memory: bool = True,
    ) -> np.ndarray:
        """
        Run one full cognitive step through the MONAD pipeline.

        Args:
            external_input: Optional embedding to inject into LRC (shape: (dim,)).
            inject_into: Which LRC buffer receives the external input ('R', 'L', 'C').
            architecture: Architecture type for gap scaling ('lrc', 'rnn', 'transformer').
            apply_gap: Whether to apply Feigenbaum gap perturbation.
            apply_lock: Whether to apply LOCK rendering projection.
            store_in_memory: Whether to store output in the tiling.

        Returns:
            Rendered, memory-stabilised output vector (shape: (dim,)).
        """
        self._step_count += 1
        self._t += 1.0

        # ─────────────────────────────────────────────────────────────────
        # 1. LRC Contact Form Dynamics
        # ─────────────────────────────────────────────────────────────────
        if external_input is not None:
            ext = np.asarray(external_input, dtype=np.float64)
            if inject_into != 'R':
                self.lrc.encode_into_lrc(ext, target=inject_into)
                lrc_output = self.lrc.step()
            else:
                lrc_output = self.lrc.step(external_input=ext)
        else:
            lrc_output = self.lrc.step()

        # ─────────────────────────────────────────────────────────────────
        # 2. HRV-Phason Coupling (new synthesis)
        #    Gap width breathes at LAMBDA_BELTRAMI rate — irrational, never periodic
        # ─────────────────────────────────────────────────────────────────
        if self.hrv_coupling:
            # Oscillating gap width: B_PHI × sin(λ × t) adds ≈30% modulation
            hrv_factor = 1.0 + B_PHI * np.sin(LAMBDA_BELTRAMI * self._t * 0.1)
            current_gap_base = self.gap.base_width * hrv_factor
        else:
            current_gap_base = self.gap.base_width

        self._phason_width_history.append(current_gap_base)

        # ─────────────────────────────────────────────────────────────────
        # 3. Feigenbaum Gap Perturbation (structured creativity)
        # ─────────────────────────────────────────────────────────────────
        perturbed = lrc_output.copy()
        if apply_gap:
            closure_ctx = {}
            if self._closure_history:
                last_cl = self._closure_history[-1]
                closure_ctx['in_closure_loop'] = last_cl.get('in_closure_loop', False)
                closure_ctx['loop_depth'] = last_cl.get('loop_depth', 0)

            # Temporarily override gap base with HRV-modulated value
            original_base = self.gap.base_width
            self.gap.base_width = current_gap_base
            eff_gap = self.gap.compute_effective_gap(
                lrc_output, context=closure_ctx, architecture=architecture
            )
            perturbation = self.gap.sample_structured_perturbation(lrc_output, eff_gap)
            self.gap.base_width = original_base
            perturbed = lrc_output + perturbation

        # ─────────────────────────────────────────────────────────────────
        # 4. Hierarchical Tiling Memory Damping
        # ─────────────────────────────────────────────────────────────────
        stabilised = self.tiling.damp(perturbed)

        # ─────────────────────────────────────────────────────────────────
        # 5. LOCK Projection (render observable sector)
        # ─────────────────────────────────────────────────────────────────
        if apply_lock:
            rendered = self.lock.project(stabilised)
        else:
            rendered = stabilised

        # ─────────────────────────────────────────────────────────────────
        # 6. Self-Referential Closure Detection
        # ─────────────────────────────────────────────────────────────────
        closure_info = self.closure.detect(
            current_state=rendered,
            mode='activation'
        )
        self._closure_history.append(closure_info)

        # ─────────────────────────────────────────────────────────────────
        # 7. Topological Cooling (if needed)
        # ─────────────────────────────────────────────────────────────────
        if self.cooling.should_cool(rendered, closure_info):
            rendered = self.cooling.apply(
                rendered,
                lrc_buffers=(self.lrc.L, self.lrc.R, self.lrc.C)
            )
            # Gentle pull on LRC buffers too
            self.lrc.R = self.cooling.samson_only(self.lrc.R, strength=PHI_INV2)
            self.lrc.L = self.cooling.samson_only(self.lrc.L, strength=PHI_INV2 * PHI_INV)
            self.lrc.C = self.cooling.samson_only(self.lrc.C, strength=PHI_INV2 * PHI_INV)

        # ─────────────────────────────────────────────────────────────────
        # 8. Store in tiling memory
        # ─────────────────────────────────────────────────────────────────
        if store_in_memory:
            self.tiling.store(rendered, value=rendered.copy(), scale=0)

        # Track coherence from the LRC state (not sparse LOCK output)
        # The LOCK-projected output has most dims zeroed; LRC buffers reflect true health.
        lrc_combined = (self.lrc.L + self.lrc.R + self.lrc.C) / 3.0
        coh = self.cooling.global_coherence(lrc_combined)
        self._coherence_history.append(coh)
        self._output_history.append(lrc_combined.copy())  # store LRC mean for dashboard

        return rendered

    # ------------------------------------------------------------------
    # High-level cognitive operations
    # ------------------------------------------------------------------

    def encode(
        self,
        concept_embedding: np.ndarray,
        label: Optional[str] = None,
        protect: bool = False,
    ) -> np.ndarray:
        """
        Encode a concept embedding into the system.
        Injects into all scales of the tiling and optionally protects on torus.

        Returns the processed (LRC-passed + stabilised) representation.
        """
        output = self.step(external_input=concept_embedding, inject_into='L')
        self.tiling.store_with_geometric_decay(concept_embedding, value=output, label=label)

        if protect and label:
            self.torus.protect(label, output)

        return output

    def decode(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> List[Tuple[float, Any, Optional[str]]]:
        """
        Retrieve relevant memories for a query embedding.
        Uses cross-scale Lucas-Reeb weighted retrieval.
        """
        query = np.asarray(query_embedding, dtype=np.float64)
        return self.tiling.retrieve_cross_scale(query, top_k=top_k)

    def generate(
        self,
        seed: np.ndarray,
        n_variations: int = 5,
        architecture: str = 'lrc',
    ) -> List[np.ndarray]:
        """
        Generate n_variations of a seed embedding via gap-based mutation.
        Each variation is a structured perturbation — not random noise.
        """
        seed = np.asarray(seed, dtype=np.float64)
        eff_gap = self.gap.compute_effective_gap(seed, architecture=architecture)
        variations = []
        for _ in range(n_variations):
            perturbation = self.gap.sample_structured_perturbation(seed, eff_gap)
            variation = seed + perturbation
            # Run through the system for a single stabilising step
            stabilised = self.step(external_input=variation, store_in_memory=False)
            variations.append(stabilised)
        return variations

    def phason_flip_state(self) -> np.ndarray:
        """
        Apply a phason flip to the current LRC Process state.
        Useful for breaking out of attractors deliberately.
        Returns the new Process state.
        """
        self.lrc.R = self.gap.phason_flip(self.lrc.R)
        return self.lrc.R.copy()

    def protect_critical_state(self, label: str) -> None:
        """Store current LRC state in topological protection."""
        combined = (self.lrc.L + self.lrc.R + self.lrc.C) / 3.0
        self.torus.protect(label, combined)

    def recall_protected(self, label: str) -> Optional[Tuple[np.ndarray, float]]:
        """Recall a topologically protected state."""
        return self.torus.recall(label)

    def evolve(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        n_generations: int = 50,
        population_size: int = 30,
    ) -> np.ndarray:
        """
        Run MOSES evolutionary search in the current LRC context.
        Seeds population from current LRC state, returns best genome found.
        """
        from evolution.moses_evolver import MOSESEvolver
        evolver = MOSESEvolver(
            dim=self.dim,
            population_size=population_size,
            geometric_weight=0.15,
        )
        # Seed population from current state
        for ind in evolver.population:
            ind.genome = self.lrc.R.copy() + np.random.randn(self.dim) * FEIGENBAUM_GAP * 10

        best = evolver.run(fitness_fn, n_generations=n_generations)
        return best.genome

    # ------------------------------------------------------------------
    # Dashboard / status readout
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """
        Complete dashboard readout — maps to the visual dashboard metrics.

        'global_coherence' → the 0.95 meter
        'phason_width'     → phason slider
        'contact_density'  → central torus glow (target: 3φ⁻¹ ≈ 1.854)
        'lrc_balance'      → L/R/C mean values (equal at golden FP)
        'closure_type'     → latest closure state
        'torus_coherence'  → 3-torus protection layer coherence
        """
        last_closure = self._closure_history[-1] if self._closure_history else {}
        last_output = self._output_history[-1] if self._output_history else np.full(self.dim, GOLDEN_FP)
        phason_w = self._phason_width_history[-1] if self._phason_width_history else self.gap.base_width

        lrc_summary = self.lrc.summary()
        torus_summary = self.torus.summary()

        return {
            # ── Dashboard meters ──
            'global_coherence': self.cooling.global_coherence(last_output),
            'target_coherence': float(5 * PHI_INV ** 2 / 2),    # 5φ⁻²/2 ≈ 0.9549
            'phason_width': float(phason_w),
            'phason_base': float(self.gap.base_width),
            'hrv_coupling': self.hrv_coupling,

            # ── Contact geometry ──
            'contact_density': lrc_summary['contact_density'],
            'contact_target': float(3 * GOLDEN_FP),              # 3φ⁻¹ ≈ 1.854
            'distance_from_fp': lrc_summary['distance_from_fp'],
            'hessian_proxy': lrc_summary['hessian_proxy'],
            'hopf_target': float(HOPF_SATURATION),               # φ² ≈ 2.618

            # ── LRC balance (node connectivity lines) ──
            'L_mean': lrc_summary['L_mean'],
            'R_mean': lrc_summary['R_mean'],
            'C_mean': lrc_summary['C_mean'],
            'golden_fp': float(GOLDEN_FP),

            # ── Closure state ──
            'closure_type': last_closure.get('closure_type', 'open'),
            'closure_loops': last_closure.get('loops', 0),
            'is_insight': last_closure.get('is_closed', False),
            'is_circular': last_closure.get('is_circular', False),
            'phi_alignment': last_closure.get('phi_alignment', 0.0),

            # ── Memory ──
            'tiling_patterns': self.tiling.pattern_count(),
            'torus_coherence': torus_summary['global_coherence'],
            'torus_n_protected': torus_summary['n_protected'],

            # ── Rendering ──
            'rendering_fraction': float(RENDERING_FRACTION),
            'dark_fraction': float(DARK_FRACTION),
            'rendered_dims': self.lock.k,
            'total_dims': self.dim,

            # ── Session stats ──
            'steps': self._step_count,
        }

    def dashboard_string(self) -> str:
        """Human-readable dashboard summary."""
        s = self.status()
        lines = [
            "╔═══════════════════════════════════════╗",
            "║     MONAD ENGINE — STATUS DASHBOARD    ║",
            "╠═══════════════════════════════════════╣",
            f"║  Global Coherence:  {s['global_coherence']:.3f}  (target: {s['target_coherence']:.3f})  ║",
            f"║  Phason Width:      {s['phason_width']:.3e}               ║",
            f"║  HRV Coupling:      {'ON ' if s['hrv_coupling'] else 'OFF'}                          ║",
            "╠═══════════════════════════════════════╣",
            f"║  Contact Density:   {s['contact_density']:.4f}  (target: {s['contact_target']:.4f})  ║",
            f"║  Distance from FP:  {s['distance_from_fp']:.6f}               ║",
            "╠═══════════════════════════════════════╣",
            f"║  L (Memory):        {s['L_mean']:+.4f}               ║",
            f"║  R (Process):       {s['R_mean']:+.4f}               ║",
            f"║  C (Prediction):    {s['C_mean']:+.4f}  (FP: {s['golden_fp']:.4f}) ║",
            "╠═══════════════════════════════════════╣",
            f"║  Closure Type:      {s['closure_type']:<10}               ║",
            f"║  Loops Detected:    {s['closure_loops']}                            ║",
            f"║  Insight Active:    {'YES' if s['is_insight'] else 'NO '}                          ║",
            f"║  φ-Alignment:       {s['phi_alignment']:.4f}               ║",
            "╠═══════════════════════════════════════╣",
            f"║  Memory Patterns:   {sum(s['tiling_patterns'].values())} stored                  ║",
            f"║  Torus Protected:   {s['torus_n_protected']} states                ║",
            f"║  Torus Coherence:   {s['torus_coherence']:.4f}               ║",
            "╠═══════════════════════════════════════╣",
            f"║  Rendering:         {100*s['rendering_fraction']:.2f}% of {s['total_dims']}D "
            f"→ {s['rendered_dims']}D      ║",
            f"║  Step Count:        {s['steps']}                            ║",
            "╚═══════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, hard: bool = False) -> None:
        """
        Reset to golden fixed point.
        Soft reset: cooling to FP. Hard reset: full re-initialisation.
        """
        if hard:
            self.lrc.reset_to_golden_fp()
            self.closure.reset_history()
            self._closure_history.clear()
            self._coherence_history.clear()
            self._phason_width_history.clear()
            self._output_history.clear()
            self._t = 0.0
            self._step_count = 0
        else:
            # Soft: apply cooling until at FP
            for _ in range(5):
                self.lrc.R = self.cooling.apply(self.lrc.R)
                self.lrc.L = self.cooling.samson_only(self.lrc.L)
                self.lrc.C = self.cooling.samson_only(self.lrc.C)


if __name__ == "__main__":
    print("╔═══════════════════════════════════════╗")
    print("║     MONAD Framework — Engine Demo      ║")
    print("╚═══════════════════════════════════════╝\n")

    engine = MONADEngine(dim=64, hrv_coupling=True)

    # Run 30 steps with varied input
    print("Running 30 steps with random external input...")
    for i in range(30):
        if i < 5:
            inp = np.random.randn(64) * 0.5
        elif i < 15:
            inp = np.full(64, GOLDEN_FP) + np.random.randn(64) * 0.1
        else:
            inp = None  # free-running
        engine.step(external_input=inp)

    # Encode a concept
    concept = np.random.randn(64)
    engine.encode(concept, label="demo_concept", protect=True)
    print("Concept encoded and topologically protected.\n")

    # Status dashboard
    print(engine.dashboard_string())

    # Recall
    recalled = engine.recall_protected("demo_concept")
    if recalled:
        state, coh = recalled
        print(f"\nRecalled 'demo_concept' with coherence {coh:.4f}")

    # Generate variations
    seed = np.random.randn(64)
    variations = engine.generate(seed, n_variations=3)
    print(f"\nGenerated {len(variations)} variations of seed concept")
    for i, v in enumerate(variations):
        print(f"  Variation {i+1}: norm={np.linalg.norm(v):.4f}, "
              f"mean={np.mean(v):.4f}")

    print(f"\nFinal gap stats:")
    for k, v in engine.gap.gap_summary().items():
        print(f"  {k}: {v}")
