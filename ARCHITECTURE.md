# MONAD Framework — Architecture (Revised May 2026)

**Status:** Implementation complete for Phases 0–5. Code verified working.
**Branch:** `claude/refine-analogies-cLOvL`

---

## Analogy Scope Clarification

This framework uses mathematical structures from the MONAD physics framework as
*functional analogies* for a cognitive architecture. Where the physics claims are
exact theorems (MONAD identity, Pi-Pentad, Golden Witt Double Cover), the code
implements them exactly. Where the biological/cosmological interpretation is
speculative (e.g., phason-HRV coupling, Sephirot overlays), the code implements
the functional structure — which is well-posed and useful — without asserting the
metaphysical claim.

**Exact mathematical structures implemented in code:**
- φψ = -1 (root identity, used throughout)
- φ⁻⁵ + 5φ⁻² = 2 (MONAD identity, asserted at import time)
- π = 5·arccos(φ/2) (Pi-Pentad, asserted at import time)
- α∧dα = (L+R+C)·ω (contact 3-form, implemented in LRCDynamics)
- Feigenbaum gap |δ_S - δ_L| ≈ 1.47×10⁻⁵ (implemented as mutation scale)
- Perfect Spine Theorem (division algebra scale hierarchy in tiling)
- Prime-dark structure (Frob_p*(α) = 0, implemented in LOCK projector)

---

## Core Principles

1. **Ordering precedes rendering** — contact structure (LRC coupling) computed before LOCK projection
2. **Geometry is first-class** — φ-scaling is in the math, not bolted on
3. **Structured chaos over pure noise** — Feigenbaum gap as mutation source
4. **Topological protection** — 3-torus for critical persistent state
5. **Cooling is mandatory** — never generate without a reset path

---

## Module Structure

```
core/
  constants.py          All φ-derived constants with verified assertions
  lrc_dynamics.py       LRC contact form engine (Memory/Process/Prediction)
  lock_projector.py     LOCK rendering projection (φ⁻⁵/2 = 4.51% rendered)

memory/
  hierarchical_tiling.py  φ-scaled geometric memory (4-scale spine hierarchy)
  torus_protection.py     3-torus topological protection for critical state

generation/
  state_dependent_gap.py  Feigenbaum gap mutation (phonon + phason modes)

detection/
  self_referential_closure.py  Closure loop detector (3-loop FP criterion)

cooling/
  topological_cooling.py  3-step cooling: Samson + Symplectic + LRC transaction

evolution/
  moses_evolver.py        MOSES evolutionary search with gap mutation + φ-fitness

core/
  quantum_chaos_controller.py  Unified controller (uses all modules above)

monad_engine.py           Top-level engine with dashboard status readout
```

---

## Layer Architecture

### Layer 0: Mathematical Substrate

**Implemented in:** `core/constants.py`

All constants derive from the morpheme set {∅, π, φ, i}. Key values:
- φ = (1+√5)/2  ≈ 1.6180
- RENDERING_FRACTION = φ⁻⁵/2 ≈ 0.04508
- FEIGENBAUM_GAP ≈ 1.47×10⁻⁵ (|δ_S - δ_L|, verified numerically)
- LAMBDA_BELTRAMI = 2π/φ ≈ 3.883 (circulation rate, irrational → stable)
- GOLDEN_FP = φ⁻¹ ≈ 0.6180 (attractor of all dynamics)

---

### Layer 1: LRC Contact Form (Memory/Process/Prediction)

**Implemented in:** `core/lrc_dynamics.py`

The contact form α = L dR + R dC + C dL is the cyclic coupling law:
- L (Memory): updated by Process buffer
- R (Process): updated by Prediction buffer (+ external input)
- C (Prediction): updated by Memory buffer (closure)

Update rule (discrete Reeb flow):
```
dL/dt = λ · tanh(R - L)   [memory tracks process]
dR/dt = λ · tanh(C - R)   [process tracks prediction]
dC/dt = λ · tanh(L - C)   [prediction tracks memory — the closing loop]
```

where λ = LAMBDA_BELTRAMI = 2π/φ (Beltrami coupling rate).

Golden fixed point: L = R = C = φ⁻¹.
At FP: contact density = mean(L+R+C) → 3φ⁻¹ ≈ 1.854 (non-degenerate contact).
The factor of 3 is required (3-torus has 3 non-contractible loops minimum for fixed point).

---

### Layer 2: LOCK Projection (Rendering)

**Implemented in:** `core/lock_projector.py`

Projects full state to observable sector: k = ceil(dim × φ⁻⁵/2) dimensions.

Three modes:
- `top_k`: keep k largest-magnitude components (dynamic, state-dependent)
- `prime_dark`: keep non-prime-indexed components only
  *Implements Frob_p*(α∧dα) = 0: prime-indexed modes are "dark"*
- `phi_harmonic`: keep components most aligned with φ-harmonic basis

For practical AI use: `top_k` is the default (fast, deterministic given state).
`prime_dark` is the most mathematically faithful to the contact geometry theorem.

---

### Layer 3: Hierarchical Tiling Memory

**Implemented in:** `memory/hierarchical_tiling.py`

Four scales mapped to the Perfect Spine Theorem (division algebra hierarchy):
```
Scale 0 → R-class  (Real,       n≡1  mod 12) — fastest,  1 buffer
Scale 1 → C-class  (Complex,    n≡5  mod 12) — 2-buffer coupling
Scale 2 → H-class  (Quaternion, n≡7  mod 12) — 4-buffer coupling
Scale 3 → O-class  (Octonion,   n≡11 mod 12) — full LRC, 8-buffer
```

Decay rate at scale k: φ^(-k) (higher scale = slower forgetting = more persistent).

Bond strength: κ = φ^(-cosine_distance × decay_rate_k × phi_strength)

Retrieval uses Lucas-Reeb series weighting across scales:
```
w_k ∝ φ^(-L_k) where L_k are Lucas prime indices (5, 7, 11, 13, ...)
```
This is the same alternating Lucas-prime structure as the Weinberg angle series.

The tiling IS the reasoning (not just storage). Holographic inference: local
patch (25% of query) → full pattern reconstruction via nearest-neighbour completion.

---

### Layer 4: 3-Torus Protection

**Implemented in:** `memory/torus_protection.py`

Three independent Beltrami-phase encodings (ω_k = k × LAMBDA_BELTRAMI):
- ω₀ = 0 (L-winding, memory direction)
- ω₁ = LAMBDA_BELTRAMI ≈ 3.883 (R-winding, process direction)
- ω₂ = 2 × LAMBDA_BELTRAMI ≈ 7.766 (C-winding, prediction direction)

Phase offsets are irrational multiples of each other → independent windings → genuine
non-contractible loops (not redundant copies of the same encoding).

Decode via per-element median → robust against single-copy corruption.
Global coherence target: 5φ⁻²/2 ≈ 0.955 (dark fraction — all dark sector engaged).

---

### Layer 5: Feigenbaum Gap Generation

**Implemented in:** `generation/state_dependent_gap.py`

Two modes of perturbation:

**Phonon (continuous):** small, structured nudge
- Amplitude: Beta(2,5) × effective_gap  (mostly small, tail allows larger)
- Direction: φ-harmonic basis projection (geometrically coherent)

**Phason (discrete):** tile flip at Lucas-prime positions
- Amplitude: φ² × effective_gap (significantly larger)
- Targets: indices = {L_n mod dim : n ∈ Lucas prime indices}

State-dependent gap width:
```
effective_gap = base_width × (1 + divergence × φ) × arch_factor × closure_factor
```
At golden FP: gap = base_width (minimal, precise).
Far from FP: gap widens to base_width × (1 + φ) ≈ 2.618 × base_width.
In circular loop: gap widens by φ² to escape.

**New synthesis (this session): HRV-Phason Coupling**
```
phason_width(t) = base_width × (1 + B_PHI × sin(LAMBDA_BELTRAMI × t × 0.1))
```
B_PHI = 2·ln(φ)/π ≈ 0.306 (dimensionless Beltrami stability constant).
The gap breathes at rate λ = 2π/φ (irrational → never periodic → always fresh).
This models HRV: biological heart rate variability driven by vagal tone
= Beltrami-phason modulation of the cognitive gap.

---

### Layer 6: Closure Detection

**Implemented in:** `detection/self_referential_closure.py`

Threshold: 3 loops (from the contact geometry fixed-point criterion).

Classification:
- `insight`: ≥3 loops + converging + φ-alignment > 0.7 → genuine fixed point
- `productive`: ≥3 loops + converging OR moderate φ-alignment → keep going
- `circular`: ≥3 loops + NOT converging + low alignment → trigger cooling
- `open`: <3 loops → normal operation

The 3-loop threshold corresponds to the three non-contractible loops of T³ required
for a non-degenerate contact structure (α∧dα ≠ 0).

---

### Layer 7: Topological Cooling

**Implemented in:** `cooling/topological_cooling.py`

Three steps (one per winding direction of T³):

1. **Samson Feedback** (L-direction): φ-weighted contraction to golden FP
   ```
   new = (1 - φ⁻¹) × state + φ⁻¹ × φ⁻¹
   ```
   φ⁻¹ ≈ 0.618 is the optimal feedback gain: minimises overshoot following
   Fibonacci convergence (each step is φ fraction closer to FP).

2. **Symplectic Collapse** (R-direction): frequency-domain low-pass at φ⁻¹ × Nyquist
   Removes phonon noise (high-frequency), preserves phason modes (low-frequency).

3. **LRC Transaction** (C-direction): restore Z₃ symmetry of contact form
   ```
   state → (1 - φ⁻²) × state + φ⁻² × mean(L, R, C)
   ```
   Restores the cyclic invariance L→R→C→L required for α to be non-degenerate.

Global coherence = 1 - cooling_urgency (the dashboard's 0.95 meter).
Target coherence: 5φ⁻²/2 ≈ 0.955 (dark fraction engaged).

---

### Layer 8: MOSES Evolutionary Search

**Implemented in:** `evolution/moses_evolver.py`

Innovations over standard MOSES:
1. **Gap mutation** replaces Gaussian noise (Beta(2,5) × effective_gap)
2. **Geometric fitness bonus** rewards φ-harmonic alignment
3. **Lucas-prime checkpoints** trigger at prime-indexed generations
4. **Phason flip diversity** at rate `phason_rate` (default 5%)
5. **Population cooling** applied every N generations

φ-weighted crossover:
```
child = φ⁻¹ × parent_a + (1-φ⁻¹) × parent_b
```
Each crossover step is itself a Fibonacci-ratio blend, converging geometrically.

---

## Data Flow

```
External Input
      │
      ▼
[LRC Contact Form] ──── tanh(R-L), tanh(C-R), tanh(L-C) ────┐
      │                                                        │
      ▼                                                        │
[HRV-Phason Gap] ─── oscillating gap width (λ=2π/φ) ─────────┤
      │                                                        │
      ▼                                                        │
[Feigenbaum Gap Mutation] ─── Beta(2,5) perturbation ─────────┤
      │                                                        │
      ▼                                                        │
[Tiling Memory Damping] ─── φ-bond-strength retrieval ────────┤
      │                                                        │
      ▼                                                        │
[LOCK Projection] ─── top-k rendering (4.51% observable) ─────┤
      │                                                        │
      ▼                                                        │
[Closure Detection] ─── 3-loop FP criterion ──────────────────┤
      │                                                        │
      ▼                                                        │
[Topological Cooling] ─── 3-step if needed ───────────────────┘
      │
      ▼
    Output (LRC Process state + coherence metrics)
```

---

## Dashboard Metric Mapping

From the visual dashboard (Grok-generated, May 2026):

| Dashboard Element      | Code Metric                          | Target Value  |
|------------------------|--------------------------------------|---------------|
| Global Coherence       | `cooling.global_coherence(lrc_mean)` | 0.955 (5φ⁻²/2)|
| Phason Slider          | `gap.base_width × hrv_factor`        | variable      |
| Central Torus (Heart)  | `lrc.contact_3form_density()`        | 3φ⁻¹ ≈ 1.854  |
| Node connectivity      | LRC balance (L, R, C means)          | all = φ⁻¹     |
| HRV waveform           | B_PHI × sin(LAMBDA_BELTRAMI × t)     | ≈ 0.306 amp   |
| Torus coherence        | `torus.global_coherence()`           | ≥ 0.95        |
| Scale 0 (Pancreas)     | R-class tiling scale                 | fastest decay |
| Scale 3 (Heart)        | O-class tiling scale, full LRC       | slowest decay |

---

## Open Connections (for future work)

- **Sephirot overlay**: Map 10 Sephirot to tiling nodes at specific scale/phi-alignment positions
- **Multi-node engine**: Multiple MONADEngine instances coupled via shared tiling (swarm architecture)
- **Financial modelling**: Use gap for regime-shift detection (Feiganary Market Engine)
- **Character gap**: Connect rendering fraction φ⁻⁵/2 to G₂(1) Verlinde vacuum 1/(φ+2)
- **Modular form**: Connect tiling retrieval weights to level-20 weight-1 newform coefficients

---

*Architecture validated against MONAD Mathematical Codex v2 (May 2026).*
*Code: github.com/agentgptsmith/monad-framework, branch claude/refine-analogies-cLOvL*
