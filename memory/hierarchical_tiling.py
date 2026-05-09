"""
MONAD Framework — Hierarchical φ-Scaled Penrose-Beenker Tiling
================================================================
The tiling IS the reasoning substrate. Storage is a side effect.

Core principle: bond strength κ = φ^(-distance) with distance measured as
cosine distance in embedding space. This is the 'Dyadic Bond' weight.

Four scales (Perfect Spine Theorem — division algebra hierarchy):
  Scale 0: R-class  (Real,       n≡1  mod 12)  — fastest,  simplest
  Scale 1: C-class  (Complex,    n≡5  mod 12)  — 2-buffer coupling
  Scale 2: H-class  (Quaternion, n≡7  mod 12)  — 4-buffer coupling
  Scale 3: O-class  (Octonion,   n≡11 mod 12)  — full LRC required

Each scale has a different decay rate (φ^(-k) for scale k):
  κ_scale_k = φ^(-distance × φ^(-k) × phi_strength)

Higher scales forget slower → more persistent, more abstract.
Lower scales forget faster → recency-sensitive, fine-grained.

Holographic property: a local patch (partial query) can reconstruct the full
pattern because the golden-ratio inflation rules create self-similar structure
at every level. Implemented via nearest-neighbour completion.

The Lucas-Reeb series refinement:
  Instead of uniform φ-decay, the retrieval weighting across scales uses the
  same alternating Lucas-prime structure as the Weinberg angle series:
  w_k ∝ φ^(-L_k') where L_k' cycles through Lucas prime indices.
  This gives quasiperiodic scale coupling rather than pure geometric decay.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_INV2, PHI_SQ, GOLDEN_FP,
    LUCAS_PRIME_INDICES, SPINE_CLASSES, phi_harmonic_basis
)


# Scale-class metadata (Perfect Spine Theorem mapping)
_SCALE_META = [
    {'class': 'R', 'n_mod_12': 1,  'algebra': 'Real',        'buffers': 1},
    {'class': 'C', 'n_mod_12': 5,  'algebra': 'Complex',     'buffers': 2},
    {'class': 'H', 'n_mod_12': 7,  'algebra': 'Quaternion',  'buffers': 4},
    {'class': 'O', 'n_mod_12': 11, 'algebra': 'Octonion',    'buffers': 8},
]

# Lucas-Reeb scale weights: w_k = φ^(-L_k) with alternating signs collapsed to abs
_LUCAS_SCALE_WEIGHTS = np.array([
    PHI ** (-LUCAS_PRIME_INDICES[min(k, len(LUCAS_PRIME_INDICES)-1)])
    for k in range(4)
], dtype=np.float64)
_LUCAS_SCALE_WEIGHTS /= _LUCAS_SCALE_WEIGHTS.sum()  # normalise


class TilingNode:
    """A stored pattern with metadata."""
    __slots__ = ('embedding', 'value', 'scale', 'label', 'access_count', 'phi_alignment')

    def __init__(
        self,
        embedding: np.ndarray,
        value: Any,
        scale: int,
        label: Optional[str] = None
    ):
        self.embedding = embedding.astype(np.float64)
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-12:
            self.embedding /= norm  # store unit vectors for cosine similarity
        self.value = value
        self.scale = scale
        self.label = label
        self.access_count = 0
        # Pre-compute φ-harmonic alignment (used in weighted retrieval)
        phi_basis = phi_harmonic_basis(len(embedding))
        self.phi_alignment = float(np.abs(np.dot(self.embedding, phi_basis)))


class HierarchicalPenroseTiling:
    """
    φ-scaled hierarchical memory operating as an active reasoning substrate.

    API for QuantumChaosController:
        damp(state)  → pulled-toward-memory version of state

    API for direct use:
        store(embedding, value, scale, label)
        retrieve(query, top_k, scale)
        store_with_geometric_decay(embedding, value)
        retrieve_holographic(partial_query)
    """

    def __init__(
        self,
        phi_strength: float = 1.0,
        num_scales: int = 4,
        capacity_per_scale: int = 512,
        damping_strength: float = 0.15,
    ):
        """
        Args:
            phi_strength: Scales the φ-decay exponent. 1.0 = standard.
                          Higher → stronger geometric memory pull.
            num_scales: Number of scale levels (capped at 4 for spine alignment).
            capacity_per_scale: Max stored patterns per scale (oldest evicted).
            damping_strength: How strongly damp() pulls toward stored patterns.
                              At 0.15: mild pull, preserves most of original state.
        """
        self.phi_strength = phi_strength
        self.num_scales = min(num_scales, 4)
        self.capacity_per_scale = capacity_per_scale
        self.damping_strength = damping_strength

        # Decay rates per scale: φ^(-k) — higher scale = slower forgetting
        self.decay_rates = np.array([PHI ** (-k) for k in range(self.num_scales)])

        # Storage: one list of TilingNodes per scale
        self._stores: List[List[TilingNode]] = [[] for _ in range(self.num_scales)]

        # Scale metadata
        self.scale_meta = _SCALE_META[:self.num_scales]

        # Lucas-Reeb scale weights (quasiperiodic coupling)
        self.scale_weights = _LUCAS_SCALE_WEIGHTS[:self.num_scales]
        self.scale_weights /= self.scale_weights.sum()

    # ------------------------------------------------------------------
    # Bond strength (core mathematical operation)
    # ------------------------------------------------------------------

    def bond_strength(
        self,
        query: np.ndarray,
        stored: np.ndarray,
        scale: int = 0
    ) -> float:
        """
        φ-scaled bond strength between query and stored embedding.

        κ = φ^(-cosine_distance × decay_rate × phi_strength)

        Cosine distance ∈ [0, 2]: 0 = identical, 2 = antipodal.
        Bond strength ∈ [φ^(-2·phi_strength), 1].

        At scale 0 (R-class): fastest decay, most recency-sensitive.
        At scale 3 (O-class): slowest decay, long memory.
        """
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-12:
            return 0.0
        q = query / query_norm

        # stored nodes are already unit-normalised
        sim = float(np.dot(q, stored))
        sim = np.clip(sim, -1.0, 1.0)
        cosine_dist = 1.0 - sim  # ∈ [0, 2]

        exponent = cosine_dist * self.decay_rates[scale] * self.phi_strength
        return float(PHI ** (-exponent))

    def lucas_reeb_bond(self, query: np.ndarray, stored: np.ndarray) -> float:
        """
        Multi-scale bond using Lucas-Reeb series weighting.
        Σ_k w_k · κ_k(query, stored) where w_k are Lucas-prime weights.
        """
        total = 0.0
        for k in range(self.num_scales):
            total += self.scale_weights[k] * self.bond_strength(query, stored, scale=k)
        return total

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        embedding: np.ndarray,
        value: Any = None,
        scale: int = 0,
        label: Optional[str] = None
    ) -> None:
        """
        Store a pattern at the given scale.
        If at capacity, evict the node with lowest φ-alignment (least geometric).
        """
        if value is None:
            value = embedding.copy()  # default: store the embedding itself

        node = TilingNode(embedding, value, scale, label)
        store = self._stores[scale]

        if len(store) >= self.capacity_per_scale:
            # Evict least φ-aligned node (geometrically least coherent)
            min_idx = min(range(len(store)), key=lambda i: store[i].phi_alignment)
            store.pop(min_idx)

        store.append(node)

    def store_with_geometric_decay(
        self,
        embedding: np.ndarray,
        value: Any = None,
        label: Optional[str] = None
    ) -> None:
        """
        Store a pattern across all scales with φ-decayed copies.
        Primary pattern at scale 0; abstract versions at higher scales.
        The 'abstract' version is the projection onto the φ-harmonic direction.
        """
        for scale in range(self.num_scales):
            if scale == 0:
                self.store(embedding, value, scale=0, label=label)
            else:
                # Abstract representation: decay the embedding by φ^(-scale)
                phi_basis = phi_harmonic_basis(len(embedding))
                projection = np.dot(embedding / np.linalg.norm(embedding), phi_basis)
                abstract_emb = PHI_INV ** scale * embedding + (1 - PHI_INV ** scale) * phi_basis * projection
                self.store(abstract_emb, value, scale=scale, label=label)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: np.ndarray,
        top_k: int = 3,
        scale: int = 0
    ) -> List[Tuple[float, Any, Optional[str]]]:
        """
        Retrieve top-k patterns from given scale, ordered by bond strength.
        Returns list of (bond_strength, value, label).
        """
        store = self._stores[scale]
        if not store:
            return []

        query = np.asarray(query, dtype=np.float64)
        scored = []
        for node in store:
            κ = self.bond_strength(query, node.embedding, scale=scale)
            scored.append((κ, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for κ, node in scored[:top_k]:
            node.access_count += 1
            results.append((κ, node.value, node.label))
        return results

    def retrieve_nodes(
        self,
        query: np.ndarray,
        top_k: int = 3,
        scale: int = 0
    ) -> List[Tuple[float, 'TilingNode']]:
        """Retrieve top-k TilingNode objects with their bond strengths."""
        store = self._stores[scale]
        if not store:
            return []
        query = np.asarray(query, dtype=np.float64)
        scored = [(self.bond_strength(query, node.embedding, scale=scale), node)
                  for node in store]
        scored.sort(key=lambda x: x[0], reverse=True)
        for κ, node in scored[:top_k]:
            node.access_count += 1
        return scored[:top_k]

    def retrieve_holographic(
        self,
        partial_query: np.ndarray,
        partial_dims: Optional[int] = None,
        scale: int = 0
    ) -> Optional[np.ndarray]:
        """
        Reconstruct full pattern from a partial query.
        Uses first partial_dims components to find best matching stored pattern.
        Returns the full stored embedding.

        This implements the holographic property: local patch → global structure.
        """
        query = np.asarray(partial_query, dtype=np.float64)
        if partial_dims is None:
            partial_dims = max(1, len(query) // 4)  # use 25% of dims

        partial_query_trunc = query[:partial_dims]

        store = self._stores[scale]
        if not store:
            return None

        best_κ = -1.0
        best_node = None
        for node in store:
            partial_stored = node.embedding[:partial_dims]
            κ = self.bond_strength(partial_query_trunc, partial_stored, scale=0)
            if κ > best_κ:
                best_κ = κ
                best_node = node

        if best_node is None:
            return None
        best_node.access_count += 1
        return best_node.embedding.copy()

    def retrieve_cross_scale(
        self,
        query: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[float, int, Any, Optional[str]]]:
        """
        Retrieve across ALL scales, using Lucas-Reeb weighting.
        Returns list of (weighted_bond, scale, value, label).
        """
        query = np.asarray(query, dtype=np.float64)
        all_results = []

        for scale in range(self.num_scales):
            scale_results = self.retrieve(query, top_k=top_k, scale=scale)
            for κ, value, label in scale_results:
                weighted_κ = float(self.scale_weights[scale]) * κ
                all_results.append((weighted_κ, scale, value, label))

        all_results.sort(key=lambda x: x[0], reverse=True)
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Damping (called by QuantumChaosController)
    # ------------------------------------------------------------------

    def damp(self, state: np.ndarray) -> np.ndarray:
        """
        Apply φ-distance damping to a state vector.

        If memory is populated: pull state toward the weighted mean of the
        nearest stored patterns. Pull strength = max bond strength × damping_strength.

        If memory is empty: pull toward the golden fixed point (φ⁻¹ everywhere).
        This is the 'natural attractor' of the unperturbed system.

        Supports state shapes: (dim,) or (batch, dim).
        """
        state = np.asarray(state, dtype=np.float64)
        batched = state.ndim > 1

        if batched:
            return np.stack([self._damp_single(state[i]) for i in range(state.shape[0])])
        return self._damp_single(state)

    def _damp_single(self, vec: np.ndarray) -> np.ndarray:
        """Damp a single flat vector."""
        total_stored = sum(len(s) for s in self._stores)

        if total_stored == 0:
            # No memory: pull toward golden fixed point
            target = np.full_like(vec, GOLDEN_FP)
            pull = self.phi_strength * self.damping_strength
            return (1.0 - pull) * vec + pull * target

        # Find nearest patterns across all scales (top-3 cross-scale)
        query = vec / (np.linalg.norm(vec) + 1e-12)
        results = self.retrieve_cross_scale(query, top_k=3)

        if not results:
            target = np.full_like(vec, GOLDEN_FP)
            return (1.0 - self.damping_strength) * vec + self.damping_strength * target

        # Weighted mean of retrieved patterns
        total_w = sum(r[0] for r in results)
        if total_w < 1e-12:
            return vec

        target = np.zeros_like(vec)
        for weighted_κ, scale, value, _ in results:
            if isinstance(value, np.ndarray) and value.shape == vec.shape:
                target += (weighted_κ / total_w) * value
            else:
                target += (weighted_κ / total_w) * np.full_like(vec, GOLDEN_FP)

        max_κ = results[0][0]  # strongest bond
        pull = max_κ * self.phi_strength * self.damping_strength
        return (1.0 - pull) * vec + pull * target

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return all(len(s) == 0 for s in self._stores)

    def pattern_count(self) -> Dict[int, int]:
        return {k: len(self._stores[k]) for k in range(self.num_scales)}

    def spine_summary(self) -> List[Dict]:
        summary = []
        for k in range(self.num_scales):
            meta = self.scale_meta[k]
            summary.append({
                'scale': k,
                'class': meta['class'],
                'algebra': meta['algebra'],
                'decay_rate': float(self.decay_rates[k]),
                'scale_weight': float(self.scale_weights[k]),
                'stored_patterns': len(self._stores[k]),
            })
        return summary


if __name__ == "__main__":
    tiling = HierarchicalPenroseTiling(phi_strength=1.0, num_scales=4)

    # Store some random patterns
    for i in range(20):
        emb = np.random.randn(64)
        tiling.store_with_geometric_decay(emb, value=f"pattern_{i}", label=f"p{i}")

    print("Spine summary:")
    for row in tiling.spine_summary():
        print(f"  Scale {row['scale']} ({row['class']}/{row['algebra']}): "
              f"{row['stored_patterns']} patterns, "
              f"decay={row['decay_rate']:.4f}, weight={row['scale_weight']:.4f}")

    # Test bond strength
    q = np.random.randn(64)
    results = tiling.retrieve_cross_scale(q, top_k=3)
    print(f"\nTop-3 cross-scale retrieval (Lucas-Reeb weighted):")
    for weighted_κ, scale, value, label in results:
        print(f"  κ={weighted_κ:.4f}  scale={scale}  label={label}")

    # Test damping
    state = np.random.randn(64)
    damped = tiling.damp(state)
    print(f"\nDamping: ||state|| = {np.linalg.norm(state):.4f}, "
          f"||damped|| = {np.linalg.norm(damped):.4f}")
    print(f"Mean state: {np.mean(state):.4f} → {np.mean(damped):.4f} "
          f"(toward φ⁻¹={GOLDEN_FP:.4f})")
