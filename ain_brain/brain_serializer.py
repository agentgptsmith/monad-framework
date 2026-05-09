"""
MONAD Framework — Brain Serializer
====================================
Converts a retrieved tiling cluster into a compact "brain card" (~300-500 tokens).

This replaces the 50,000-token graph metadata system. The geometry encodes the
relationships; the card only stores what can't be reconstructed from geometry.

Card anatomy:
  - Header: query, coherence, contact density, gap-to-FP
  - Cluster: scale distribution, φ-span (how spread the cluster is in φ-space)
  - Nodes: top-N by bond strength, with Lucas-prime addresses
  - Bridges: strongest cross-scale connections (new information not in node list)
  - Patterns: emergent patterns detected from cluster geometry (computed, not stored)

Address format: {SCALE_CLASS}.{LUCAS_IDX}  e.g. O.11 = Octonion scale, L-index 11
  - Scale class from Perfect Spine Theorem (R/C/H/O)
  - Lucas index from dominant frequency in φ-harmonic spectrum
  These are geometric coordinates, not arbitrary IDs.

Token budget: target ≤ 500 tokens total (vs 50,000 in the old graph system).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_INV2, GOLDEN_FP,
    LAMBDA_BELTRAMI, LUCAS_PRIME_INDICES
)


_SCALE_LABELS = ['R', 'C', 'H', 'O']
_SCALE_ALGEBRAS = ['Real', 'Complex', 'Quaternion', 'Octonion']
_SCALE_MOD12 = [1, 5, 7, 11]


@dataclass
class ClusterNode:
    """A single node in a retrieved cluster."""
    label: str
    scale: int
    bond: float            # bond strength to query, ∈ [0, 1]
    phi_alignment: float   # alignment with φ-harmonic basis, ∈ [0, 1]
    value: object          # the stored value (text, embedding, or dict)
    embedding: np.ndarray

    @property
    def address(self) -> str:
        """Geometric address: {scale_class}.{dominant_lucas_index}."""
        scale_char = _SCALE_LABELS[self.scale]
        lucas_idx = _dominant_lucas_index(self.embedding)
        return f"{scale_char}.{lucas_idx}"

    @property
    def scale_label(self) -> str:
        return _SCALE_LABELS[self.scale]


@dataclass
class BrainCard:
    """Structured output of a cluster serialization."""
    query: str
    coherence: float
    contact_density: float
    gap_to_fp: float
    phi_span: float
    nodes: List[ClusterNode]
    bridges: List[Tuple[ClusterNode, ClusterNode, float]]   # (a, b, bond)
    patterns: List[str]
    scale_counts: dict = field(default_factory=dict)

    def render(self, max_nodes: int = 10, max_bridges: int = 4) -> str:
        """Render to compact string form — the actual brain card."""
        lines = _render_card(self, max_nodes, max_bridges)
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()

    def token_estimate(self) -> int:
        """Rough token estimate (4 chars ≈ 1 token)."""
        return len(self.render()) // 4


class BrainSerializer:
    """
    Converts a cluster of (node, bond_strength) pairs into a BrainCard.

    Usage:
        ser = BrainSerializer()
        card = ser.serialize(query="attention", cluster=retrieved_nodes)
        print(card.render())
    """

    def __init__(self, max_nodes: int = 10, max_bridges: int = 4):
        self.max_nodes = max_nodes
        self.max_bridges = max_bridges

    def serialize(
        self,
        query: str,
        cluster: List[Tuple[object, float, int]],  # (TilingNode, bond, scale)
        lrc_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> BrainCard:
        """
        Args:
            query: The original query string.
            cluster: List of (TilingNode, bond_strength, scale_override) tuples.
                     scale_override = -1 → use node.scale.
            lrc_state: (L, R, C) arrays for coherence computation.
                       If None, coherence estimated from cluster geometry.

        Returns:
            BrainCard — render() for the string form.
        """
        if not cluster:
            return BrainCard(
                query=query, coherence=0.0, contact_density=0.0,
                gap_to_fp=1.0, phi_span=0.0, nodes=[], bridges=[], patterns=[],
                scale_counts={'R': 0, 'C': 0, 'H': 0, 'O': 0}
            )

        nodes = _build_cluster_nodes(cluster, self.max_nodes)
        coherence, contact_density, gap_to_fp = _compute_geometry(nodes, lrc_state)
        phi_span = _phi_span(nodes)
        scale_counts = _count_scales(nodes)
        bridges = _find_bridges(nodes, self.max_bridges)
        patterns = _detect_patterns(nodes, coherence, phi_span)

        return BrainCard(
            query=query,
            coherence=coherence,
            contact_density=contact_density,
            gap_to_fp=gap_to_fp,
            phi_span=phi_span,
            nodes=nodes,
            bridges=bridges,
            patterns=patterns,
            scale_counts=scale_counts,
        )

    def serialize_empty(self, query: str) -> BrainCard:
        return BrainCard(
            query=query, coherence=0.0, contact_density=0.0,
            gap_to_fp=1.0, phi_span=0.0, nodes=[], bridges=[], patterns=[],
            scale_counts={'R': 0, 'C': 0, 'H': 0, 'O': 0}
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dominant_lucas_index(embedding: np.ndarray) -> int:
    """
    Find the dominant Lucas-prime index in the embedding's φ-harmonic spectrum.

    Project embedding onto cos(k × 2π/φ) for k in LUCAS_PRIME_INDICES.
    The index with the highest projection amplitude is the "address component".
    Returns the Lucas-prime number (e.g. 5, 7, 11, 13...) not the list index.
    """
    n = len(embedding)
    if n == 0:
        return LUCAS_PRIME_INDICES[0]

    best_idx = LUCAS_PRIME_INDICES[0]
    best_proj = -1.0

    for lp in LUCAS_PRIME_INDICES[:8]:
        # Beltrami eigenfunction at frequency lp × 2π/φ
        freqs = np.arange(n, dtype=np.float64)
        basis = np.cos(freqs * (float(lp) * 2.0 * np.pi / PHI))
        norm = np.linalg.norm(basis)
        if norm < 1e-12:
            continue
        proj = abs(float(np.dot(embedding, basis / norm)))
        if proj > best_proj:
            best_proj = proj
            best_idx = lp

    return best_idx


def _build_cluster_nodes(
    cluster: List[Tuple[object, float, int]],
    max_nodes: int
) -> List[ClusterNode]:
    """Convert raw (TilingNode, bond, scale_override) tuples to ClusterNodes."""
    items = []
    for item in cluster:
        raw_node, bond, scale_override = item
        scale = scale_override if scale_override >= 0 else raw_node.scale
        label = getattr(raw_node, 'label', None) or f"node@{_SCALE_LABELS[scale]}"
        embedding = np.array(raw_node.embedding, dtype=np.float64)
        if np.linalg.norm(embedding) < 1e-12:
            embedding = np.ones(max(1, len(embedding))) / max(1, len(embedding)) ** 0.5

        phi_aln = getattr(raw_node, 'phi_alignment', 0.0)
        value = getattr(raw_node, 'value', None)
        items.append(ClusterNode(
            label=str(label),
            scale=int(np.clip(scale, 0, 3)),
            bond=float(np.clip(bond, 0.0, 1.0)),
            phi_alignment=float(phi_aln),
            value=value,
            embedding=embedding,
        ))

    # Sort by bond strength descending, trim
    items.sort(key=lambda n: n.bond, reverse=True)
    return items[:max_nodes]


def _compute_geometry(
    nodes: List[ClusterNode],
    lrc_state: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[float, float, float]:
    """
    Returns (coherence, contact_density, gap_to_fp).

    coherence:       how close the cluster centroid mean(|x|) is to GOLDEN_FP
    contact_density: proxy for α∧dα = (L+R+C)·ω; uses cluster mean φ-alignment
    gap_to_fp:       |mean(|centroid|) - φ⁻¹|
    """
    if not nodes:
        return 0.0, 0.0, 1.0

    if lrc_state is not None:
        L, R, C = lrc_state
        combined = (np.mean(np.abs(L)) + np.mean(np.abs(R)) + np.mean(np.abs(C))) / 3.0
        contact_density = float(
            np.mean(np.abs(L)) + np.mean(np.abs(R)) + np.mean(np.abs(C))
        )
    else:
        # Approximate from cluster embeddings
        all_embs = np.stack([n.embedding[:min(len(n.embedding), 64)] for n in nodes])
        combined = float(np.mean(np.abs(all_embs)))
        contact_density = combined * 3.0

    gap_to_fp = abs(combined - GOLDEN_FP)
    # Coherence: 1 - (gap / max_possible_gap)   max gap ≈ φ⁻¹ (from 0 to φ⁻¹)
    coherence = float(np.clip(1.0 - gap_to_fp / GOLDEN_FP, 0.0, 1.0))
    return coherence, contact_density, gap_to_fp


def _phi_span(nodes: List[ClusterNode]) -> float:
    """
    φ-span: how spread out the cluster is in φ-harmonic space.
    Measured as std of bond strengths — tight cluster = low span, diverse = high.
    """
    if len(nodes) < 2:
        return 0.0
    bonds = [n.bond for n in nodes]
    return float(np.std(bonds))


def _count_scales(nodes: List[ClusterNode]) -> dict:
    counts = {'R': 0, 'C': 0, 'H': 0, 'O': 0}
    for n in nodes:
        counts[_SCALE_LABELS[n.scale]] += 1
    return counts


def _find_bridges(
    nodes: List[ClusterNode],
    max_bridges: int
) -> List[Tuple[ClusterNode, ClusterNode, float]]:
    """
    Find strongest cross-scale bonds — the 'bridges'.
    Only cross-scale pairs qualify (same-scale connections are expected).
    Bond strength = φ^(-cosine_distance).
    """
    bridges = []
    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            if a.scale == b.scale:
                continue  # same-scale = not a bridge
            # Align dimensions
            dim = min(len(a.embedding), len(b.embedding))
            if dim < 1:
                continue
            ae = a.embedding[:dim]
            be = b.embedding[:dim]
            norm_a = np.linalg.norm(ae)
            norm_b = np.linalg.norm(be)
            if norm_a < 1e-12 or norm_b < 1e-12:
                continue
            cos_sim = float(np.dot(ae / norm_a, be / norm_b))
            cos_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)
            bond = float(PHI ** (-cos_dist))
            bridges.append((a, b, bond))

    bridges.sort(key=lambda x: x[2], reverse=True)
    return bridges[:max_bridges]


def _detect_patterns(
    nodes: List[ClusterNode],
    coherence: float,
    phi_span: float
) -> List[str]:
    """
    Detect emergent geometric patterns in the cluster.
    These are computed at runtime — not stored anywhere — which is the point.
    """
    patterns = []

    if not nodes:
        return patterns

    # Pattern 1: φ-harmonic concentration
    high_phi = [n for n in nodes if n.phi_alignment > PHI_INV]
    if len(high_phi) >= 3:
        patterns.append(f"phi-harmonic: {len(high_phi)} nodes above φ⁻¹ threshold")

    # Pattern 2: Scale diversity
    scales_present = set(n.scale for n in nodes)
    if len(scales_present) >= 3:
        patterns.append(f"multi-scale: {len(scales_present)}/4 scales represented")
    elif len(scales_present) == 1:
        single = _SCALE_LABELS[list(scales_present)[0]]
        patterns.append(f"scale-concentrated: {single}-class cluster")

    # Pattern 3: Coherence classification
    if coherence > 0.95:
        patterns.append("at-golden-FP: cluster near fixed point")
    elif coherence < 0.3:
        patterns.append("far-from-FP: cluster needs cooling")

    # Pattern 4: Tight vs sparse
    if phi_span < 0.05:
        patterns.append("tight: low φ-span, homogeneous cluster")
    elif phi_span > 0.25:
        patterns.append("sparse: high φ-span, heterogeneous cluster")

    # Pattern 5: O-class dominant (most abstract, slowest decay)
    o_nodes = [n for n in nodes if n.scale == 3]
    if o_nodes and o_nodes[0].bond > 0.8:
        patterns.append("octonion-anchored: deepest abstract match at O-scale")

    return patterns


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------

def _render_card(card: BrainCard, max_nodes: int, max_bridges: int) -> List[str]:
    """Render a BrainCard to a list of lines."""
    W = 60  # card width (chars)
    lines = []

    def rule(char='─'):
        return char * W

    def pad(s: str, width: int = W) -> str:
        return s[:width].ljust(width)

    # Header
    lines.append('┌' + rule('─') + '┐')
    q = card.query[:48] if len(card.query) > 48 else card.query
    lines.append('│  ' + pad(f"MONAD BRAIN CARD  ·  {q}", W - 2) + '│')
    lines.append('│  ' + pad(
        f"coherence: {card.coherence:.3f}  "
        f"contact: {card.contact_density:.3f}  "
        f"Δφ: {card.gap_to_fp:.4f}", W - 2
    ) + '│')
    lines.append('├' + rule('─') + '┤')

    # Cluster summary
    sc = card.scale_counts
    dist_str = f"R:{sc.get('R',0)} C:{sc.get('C',0)} H:{sc.get('H',0)} O:{sc.get('O',0)}"
    lines.append('│  ' + pad(
        f"CLUSTER  [{dist_str}]  φ-span: {card.phi_span:.3f}", W - 2
    ) + '│')

    # Nodes
    lines.append('│  ' + pad('── nodes ──', W - 2) + '│')
    shown_nodes = card.nodes[:max_nodes]
    for n in shown_nodes:
        star = '✦' if n.phi_alignment > PHI_INV else ' '
        addr = n.address
        label = n.label[:24] if len(n.label) > 24 else n.label
        row = f"  {addr:<8} {label:<26} κ={n.bond:.3f} {star}"
        lines.append('│' + pad(row, W) + '│')

    if len(card.nodes) > max_nodes:
        lines.append('│  ' + pad(f"  … +{len(card.nodes) - max_nodes} more", W - 2) + '│')

    # Bridges
    if card.bridges:
        lines.append('│  ' + pad('── bridges ──', W - 2) + '│')
        seen_addr_pairs: set = set()
        for a, b, bond in card.bridges[:max_bridges]:
            addr_pair = (a.address, b.address)
            if addr_pair in seen_addr_pairs:
                # Disambiguate with short label
                a_id = f"{a.address}:{a.label[:6]}"
                b_id = f"{b.address}:{b.label[:6]}"
            else:
                a_id, b_id = a.address, b.address
            seen_addr_pairs.add(addr_pair)
            pair = f"  {a_id} ↔ {b_id}"
            algebra_pair = f"[{_SCALE_ALGEBRAS[a.scale][:4]}↔{_SCALE_ALGEBRAS[b.scale][:4]}]"
            row = f"{pair:<26} {bond:.3f}  {algebra_pair}"
            lines.append('│' + pad(row, W) + '│')

    # Patterns
    if card.patterns:
        lines.append('│  ' + pad('── patterns ──', W - 2) + '│')
        for p in card.patterns:
            lines.append('│  ' + pad(f"  ∿ {p}", W - 2) + '│')

    lines.append('└' + rule('─') + '┘')
    return lines
