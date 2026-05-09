# Memory System - φ-Scaled Penrose-Beenker Tiling + Topological Protection

## Overview

The memory system has two tightly coupled layers:

1. **Geometric Layer**: φ-scaled Penrose-Beenker tiling with distance-weighted Dyadic Bonds
2. **Topological Layer**: 3-torus non-local storage with loop-based protection

## Geometric Layer: Penrose-Beenker Tiling

- Aperiodic tiling with golden ratio inflation rules
- Local patches are holographic (contain global information)
- Relevance / bond strength decays as `kappa = phi ** (-distance)`
- Naturally compresses irrelevant or distant concepts
- Confirmed working in practice (user verification)

**Key Properties:**
- High information density without periodicity
- Built-in error resistance (local defects don't propagate globally)
- Natural forgetting mechanism aligned with φ-scaling

## Topological Layer: 3-Torus Protection

- Logical information mapped to non-contractible loops on a 3-torus
- Physical node decoherence does not destroy logical state
- Protection equation: `|psi_logical> = C^{-1} U C |psi_physical>`
- Provides quantum-like error correction without requiring full quantum hardware

## Phase Space Gap (Deepened)

The **Dual-Class Feigenbaum gap** is the primary source of structured novelty.

**Two Limits:**
- δ_L = π + arctan(e^π)          → Observer-free geometric limit
- δ_S = τ - φ + φ^(-10)/2     → Observer-entangled algebraic limit

**The Gap:**
The small mathematical discrepancy between these two limits is where "entropic exhaust" (novel structure) is squeezed out. This is not uniform noise — it is *structured chaos* already biased toward the geometric invariants of the system (φ-scaling and topological stability).

**Mutation Implication:**
When generating variation (for MOSES or other evolutionary processes), sampling from this gap produces mutations that are:
- More likely to be compressible in the Penrose-Beenker tiling
- More likely to be stable under topological protection
- Naturally aligned with the golden ratio constraints of the substrate

This is the core of "Feigenbaum gap mutation mechanics".

## Interaction Between Layers

- New structures generated via the gap are first refined (MOSES-style)
- Then written to the tiling with φ-distance weighting
- Critical logical state is additionally protected on the 3-torus
- Retrieval uses geometric decay + holographic properties

## Implementation Notes for Sonnet

- Make the tiling the default storage for conceptual / relational data
- Use the 3-torus layer for high-value persistent state that must survive resets
- Expose `store_with_geometric_decay()` and `retrieve_holographic()` as primary APIs
- The gap sampling should be a first-class function (`sample_from_feigenbaum_gap()`)

---

*This is the working memory specification.*