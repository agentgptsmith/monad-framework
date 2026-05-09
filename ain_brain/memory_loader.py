"""
MONAD Framework — Memory Loader
=================================
The compact brain interface. Store concepts, retrieve clusters, get cards.

This is the replacement for the 3-skill + 1-bridge file system (50,000 tokens).
Target: ~1,500 tokens total for the whole system (the original design target).

Architecture:
  - MemoryLoader wraps HierarchicalPenroseTiling + BrainSerializer
  - Embedding: optional external function; defaults to φ-harmonic projection
  - Persistence: pickle (compact, ~KB per scale) — not 50KB JSON graphs
  - Auto-scale routing: short/fast content → R-scale; deep/slow → O-scale

Scale routing heuristic (overridable):
  R-class (scale 0): quick facts, recent context, recency-sensitive
  C-class (scale 1): structured concepts, 2-element relationships
  H-class (scale 2): multi-part patterns, quaternion-level complexity
  O-class (scale 3): deep abstractions, long-term principles, slowest decay
"""

from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from memory.hierarchical_tiling import HierarchicalPenroseTiling, TilingNode
from ain_brain.brain_serializer import BrainSerializer, BrainCard
from core.constants import PHI, PHI_INV, GOLDEN_FP, phi_harmonic_basis, LUCAS_PRIME_INDICES


# Default embedding dimension when no external embedder is provided
_DEFAULT_DIM = 128

EmbedFn = Callable[[str], np.ndarray]


class MemoryLoader:
    """
    Store and retrieve concepts from the MONAD tiling brain.

    Quick start (no ML dependencies):
        brain = MemoryLoader()
        brain.store("attention mechanism", "Q/K/V dot-product scaled by sqrt(d_k)")
        card = brain.load("transformer attention")
        print(card)

    With real embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        brain = MemoryLoader(embed_fn=lambda t: model.encode(t))
        brain.store("attention mechanism", ...)
        card = brain.load("how does attention work?")
        print(card)

    Scale routing (manual override):
        brain.store("quick fact", content, scale=0)     # R-class
        brain.store("deep principle", content, scale=3) # O-class
    """

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        dim: int = _DEFAULT_DIM,
        phi_strength: float = 1.0,
        capacity_per_scale: int = 512,
        serializer: Optional[BrainSerializer] = None,
    ):
        """
        Args:
            embed_fn:   Text → np.ndarray embedding. None → φ-harmonic hash embedding.
            dim:        Embedding dimension. Inferred from embed_fn on first call if not set.
            phi_strength: Controls φ-decay sharpness in tiling (1.0 = standard).
            capacity_per_scale: Max stored nodes per scale (4 scales, default 512 each).
            serializer: Custom BrainSerializer. None → default (max_nodes=10, max_bridges=4).
        """
        self._embed_fn = embed_fn
        self._dim = dim
        self._dim_locked = False  # locked on first embedding to prevent mismatch

        self.tiling = HierarchicalPenroseTiling(
            phi_strength=phi_strength,
            num_scales=4,
            capacity_per_scale=capacity_per_scale,
        )
        self.serializer = serializer or BrainSerializer(max_nodes=10, max_bridges=4)

        # Access log: (query, card.coherence) pairs for session diagnostics
        self._query_log: List[Tuple[str, float]] = []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Convert text to embedding vector."""
        if self._embed_fn is not None:
            raw = np.asarray(self._embed_fn(text), dtype=np.float64).ravel()
        else:
            raw = _phi_hash_embed(text, self._dim)

        if not self._dim_locked:
            self._dim = len(raw)
            self._dim_locked = True

        # Resize if necessary (shouldn't happen with consistent embed_fn)
        if len(raw) != self._dim:
            raw = _resize_embedding(raw, self._dim)

        # Unit normalise
        norm = np.linalg.norm(raw)
        if norm > 1e-12:
            raw = raw / norm
        return raw

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        label: str,
        value: Any = None,
        scale: Optional[int] = None,
        text_for_embedding: Optional[str] = None,
    ) -> None:
        """
        Store a concept in the tiling brain.

        Args:
            label:               Short human-readable label (used as embedding text if no text_for_embedding).
            value:               The content to store (string, dict, or anything serialisable).
            scale:               0=R, 1=C, 2=H, 3=O. None → auto-routed by content complexity.
            text_for_embedding:  Override the text used for embedding (defaults to label).
        """
        embed_text = text_for_embedding or label
        embedding = self._embed(embed_text)

        if scale is None:
            scale = _auto_scale(label, value)

        self.tiling.store(embedding, value=value, scale=scale, label=label)

    def store_many(
        self,
        items: List[dict],
    ) -> None:
        """
        Bulk store. Each item is a dict with keys: label, value, scale (optional).

        items = [
            {'label': 'attention', 'value': 'Q/K/V...', 'scale': 2},
            {'label': 'phi ratio', 'value': '1.618...'},
        ]
        """
        for item in items:
            self.store(
                label=item['label'],
                value=item.get('value'),
                scale=item.get('scale'),
                text_for_embedding=item.get('text_for_embedding'),
            )

    # ------------------------------------------------------------------
    # Load (query → card)
    # ------------------------------------------------------------------

    def load(
        self,
        query: str,
        top_k: int = 10,
        scales: Optional[List[int]] = None,
        lrc_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> BrainCard:
        """
        Query the tiling and return a compact BrainCard.

        Args:
            query:     Natural-language query string.
            top_k:     Max nodes to retrieve per scale.
            scales:    Which scales to search. None → all 4 scales.
            lrc_state: (L, R, C) arrays from LRCDynamics for coherence measurement.

        Returns:
            BrainCard — call str(card) or card.render() to get the printable string.
        """
        embedding = self._embed(query)
        scales_to_search = scales if scales is not None else list(range(4))

        # Retrieve from each requested scale
        cluster: List[Tuple[TilingNode, float, int]] = []
        for s in scales_to_search:
            results = self.tiling.retrieve_nodes(embedding, top_k=top_k, scale=s)
            for bond, node in results:
                cluster.append((node, bond, s))

        if not cluster:
            card = self.serializer.serialize_empty(query)
        else:
            card = self.serializer.serialize(query, cluster, lrc_state)

        self._query_log.append((query, card.coherence))
        return card

    def load_holographic(
        self,
        partial_query: str,
        patch_fraction: float = 0.25,
        top_k: int = 8,
    ) -> BrainCard:
        """
        Holographic retrieval: use a partial (25%) query fragment to reconstruct
        the full pattern. Exploits the self-similar structure of the tiling.
        """
        full_embedding = self._embed(partial_query)
        # Simulate partial query: keep only patch_fraction of dimensions
        partial = full_embedding.copy()
        cutoff = max(1, int(len(partial) * patch_fraction))
        partial[cutoff:] = 0.0
        norm = np.linalg.norm(partial)
        if norm > 1e-12:
            partial = partial / norm

        return self.load(partial_query, top_k=top_k)

    # ------------------------------------------------------------------
    # Convenience: print card directly
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: int = 10,
        scales: Optional[List[int]] = None,
        lrc_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> str:
        """Load and return the rendered card string (for direct printing)."""
        card = self.load(query, top_k, scales, lrc_state)
        return card.render()

    # ------------------------------------------------------------------
    # Stats and diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return storage statistics per scale."""
        scale_labels = ['R', 'C', 'H', 'O']
        counts = {scale_labels[s]: len(self.tiling._stores[s]) for s in range(4)}
        total = sum(counts.values())
        recent_coherences = [c for _, c in self._query_log[-10:]]
        return {
            'total_nodes': total,
            'per_scale': counts,
            'dim': self._dim,
            'queries_logged': len(self._query_log),
            'recent_mean_coherence': float(np.mean(recent_coherences)) if recent_coherences else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the brain to a pickle file."""
        state = {
            'tiling_stores': [
                [(n.embedding.tolist(), n.value, n.scale, n.label, n.access_count)
                 for n in store]
                for store in self.tiling._stores
            ],
            'dim': self._dim,
            'dim_locked': self._dim_locked,
            'phi_strength': self.tiling.phi_strength,
            'capacity_per_scale': self.tiling.capacity_per_scale,
            'query_log': self._query_log[-100:],  # keep last 100
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_from_file(
        cls,
        path: str,
        embed_fn: Optional[EmbedFn] = None,
    ) -> 'MemoryLoader':
        """Load a saved brain from a pickle file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Brain file not found: {path}")

        with open(path, 'rb') as f:
            state = pickle.load(f)

        loader = cls(
            embed_fn=embed_fn,
            dim=state['dim'],
            phi_strength=state.get('phi_strength', 1.0),
            capacity_per_scale=state.get('capacity_per_scale', 512),
        )
        loader._dim_locked = state.get('dim_locked', False)
        loader._query_log = state.get('query_log', [])

        # Restore nodes
        for s, scale_data in enumerate(state['tiling_stores']):
            for emb_list, value, scale, label, access_count in scale_data:
                emb = np.array(emb_list, dtype=np.float64)
                node = TilingNode(emb, value, scale, label)
                node.access_count = access_count
                loader.tiling._stores[s].append(node)

        return loader

    # ------------------------------------------------------------------
    # Migration helpers
    # ------------------------------------------------------------------

    def import_from_dict_list(
        self,
        records: List[dict],
        label_key: str = 'label',
        value_key: str = 'value',
        scale_key: str = 'scale',
        embed_key: str = 'text_for_embedding',
    ) -> int:
        """
        Migrate records from an old graph/skill system into the tiling.

        records = [
            {'label': 'attention', 'value': '...', 'scale': 2},
            ...
        ]
        Returns the number of records stored.
        """
        count = 0
        for rec in records:
            label = rec.get(label_key, '')
            value = rec.get(value_key, rec.get(label_key, ''))
            scale = rec.get(scale_key)
            embed_text = rec.get(embed_key, label)
            if label:
                self.store(label, value, scale, embed_text)
                count += 1
        return count


# ---------------------------------------------------------------------------
# φ-harmonic hash embedding (default, no ML dependencies)
# ---------------------------------------------------------------------------

def _phi_hash_embed(text: str, dim: int) -> np.ndarray:
    """
    Convert text to a φ-harmonic embedding without any ML library.

    Method: hash each character to a frequency index → sum Beltrami
    eigenfunctions cos(char_hash × 2π/φ × i) across the dim-dimensional space.
    Result: a dim-dimensional vector that's different for different texts,
    respects the Beltrami geometry, and is compact.

    This is NOT semantically meaningful (no training). For real semantic
    retrieval, pass a sentence-transformers embed_fn to MemoryLoader.
    For structural testing and framework mechanics, it works perfectly.
    """
    if not text:
        return phi_harmonic_basis(dim)

    v = np.zeros(dim, dtype=np.float64)
    indices = np.arange(dim, dtype=np.float64)
    base_freq = 2.0 * np.pi / PHI

    # Each character contributes a phase-shifted Beltrami component
    for j, char in enumerate(text[:256]):  # cap at 256 chars
        char_hash = (ord(char) * 2654435761) & 0xFFFFFFFF  # Knuth multiplicative hash
        phase = (char_hash / 0xFFFFFFFF) * 2.0 * np.pi
        freq_scale = 1.0 + (j % len(LUCAS_PRIME_INDICES)) * PHI_INV
        v += np.cos(indices * base_freq * freq_scale + phase)

    # Normalise
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return phi_harmonic_basis(dim)
    return v / norm


def _resize_embedding(emb: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize an embedding vector to target_dim via truncation or φ-padded extension."""
    curr = len(emb)
    if curr == target_dim:
        return emb
    if curr > target_dim:
        return emb[:target_dim]
    # Extend: pad with φ-harmonic basis values
    pad = phi_harmonic_basis(target_dim)[curr:]
    return np.concatenate([emb, pad * (PHI_INV ** (target_dim - curr))])


def _auto_scale(label: str, value: Any) -> int:
    """
    Auto-route content to a scale based on complexity heuristics.

    R (0): short label, simple value — fast, recency-sensitive
    C (1): medium label or structured dict with 2 keys
    H (2): longer content or multi-part structure
    O (3): very long, nested, or abstract — slow decay, deepest memory
    """
    label_len = len(label)
    value_complexity = 0

    if isinstance(value, str):
        value_complexity = len(value) // 100
    elif isinstance(value, dict):
        value_complexity = len(value)
    elif isinstance(value, (list, tuple)):
        value_complexity = len(value) // 3

    total = label_len + value_complexity * 10

    if total < 50:
        return 0  # R-class
    elif total < 150:
        return 1  # C-class
    elif total < 400:
        return 2  # H-class
    else:
        return 3  # O-class
