"""
MONAD Framework — Self-Referential Closure Detector
=====================================================
Detects when the system has formed topological fixed points (genuine insight)
vs. circular attractors (productive loops) vs. degenerate loops (stuck).

The criterion comes from contact geometry:
    A topological fixed point requires ≥ 3 non-contractible loops on T³.
    At the golden FP, α∧dα = 3φ⁻¹·ω ≠ 0 — the factor of 3 is not accidental.
    It's the minimum loop count needed for the contact structure to be non-degenerate.

CLOSURE_THRESHOLD = 3

Three closure types:
    'insight'    — ≥3 loops AND state converging (Cauchy criterion met)
                   → genuine fixed point, contact structure active
    'productive' — ≥3 loops AND state NOT yet converged but trending inward
                   → keep going, don't cool yet
    'circular'   — ≥3 loops AND state oscillating (no convergence trend)
                   → trigger cooling / phason flip to escape
    'open'       — <3 loops
                   → continue normally

Two detection modes:
    'activation' — cosine similarity of state vectors in rolling history
                   Detects repeated activation patterns (the system is revisiting
                   the same region of state space).
    'trace'      — string/token pattern matching in reasoning traces
                   Detects repeated reasoning steps in LLM output.
    'both'       — both modes, result is union

The Dabrowski OE analogy: intellectual OE creates productive loops (deep dives).
The detector distinguishes productive loops (generating new depth each pass) from
degenerate ones (repeating identical steps). The 3-loop threshold is the difference.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple
from core.constants import PHI, PHI_INV, PHI_SQ, GOLDEN_FP

ModeType = Literal['activation', 'trace', 'both']
ClosureType = Literal['insight', 'productive', 'circular', 'open']


class SelfReferentialClosureDetector:
    """
    Detects self-referential closure loops in state space or reasoning traces.

    When 3+ closure loops are detected AND state is converging → genuine insight.
    When 3+ loops detected AND state is oscillating → trigger cooling.

    The detector maintains a rolling history of states (activation mode) and/or
    reasoning trace tokens (trace mode).
    """

    CLOSURE_THRESHOLD: int = 3    # minimum loops for fixed-point status

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        window: int = 20,
        convergence_window: int = 5,
        convergence_tolerance: float = 1e-3,
    ):
        """
        Args:
            similarity_threshold: Cosine similarity above which two states are
                                  considered 'the same' (a loop).
            window: Rolling history length for activation comparison.
            convergence_window: Number of recent steps to check for Cauchy convergence.
            convergence_tolerance: Max acceptable step-to-step change for convergence.
        """
        self.similarity_threshold = similarity_threshold
        self.window = window
        self.convergence_window = convergence_window
        self.convergence_tolerance = convergence_tolerance

        # Rolling history buffers
        self._state_history: deque = deque(maxlen=window)
        self._trace_history: deque = deque(maxlen=window)

        # Running diagnostics
        self._total_detections = 0
        self._loop_type_counts: Dict[str, int] = {
            'insight': 0, 'productive': 0, 'circular': 0, 'open': 0
        }

    # ------------------------------------------------------------------
    # Main detection interface
    # ------------------------------------------------------------------

    def detect(
        self,
        current_state: np.ndarray,
        history: Optional[List[Any]] = None,
        mode: ModeType = 'both',
        reasoning_trace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect closure loops and classify the closure type.

        Args:
            current_state: Current activation state (numpy array).
            history: Optional external history list (numpy arrays or states).
                     If None, uses internal rolling buffer.
            mode: 'activation', 'trace', or 'both'.
            reasoning_trace: Optional string of recent reasoning text (for 'trace' mode).

        Returns:
            Dict with:
                'loops': int — number of detected closure loops
                'closure_type': ClosureType — 'insight'/'productive'/'circular'/'open'
                'is_closed': bool — True if genuine fixed point (≥3 loops + converging)
                'is_circular': bool — True if degenerate loop (≥3 loops + oscillating)
                'in_closure_loop': bool — True if any kind of loop (3+)
                'loop_depth': int — how many loop-detection cycles in a row
                'convergence_trend': float — step-to-step distance trend (negative=converging)
                'phi_alignment': float — current state's alignment with golden FP
                'contact_3form_proxy': float — proxy for α∧dα density
        """
        current_state = np.asarray(current_state, dtype=np.float64)
        flat = current_state.flatten()

        # Update internal history
        self._state_history.append(flat.copy())
        if reasoning_trace:
            self._trace_history.append(reasoning_trace)

        # Build history to use
        if history is not None:
            hist_states = [np.asarray(h, dtype=np.float64).flatten()
                           for h in history if hasattr(h, '__len__')]
        else:
            hist_states = list(self._state_history)[:-1]  # exclude current

        # Detection
        activation_loops = 0
        trace_loops = 0

        if mode in ('activation', 'both'):
            activation_loops = self._count_activation_loops(flat, hist_states)

        if mode in ('trace', 'both') and reasoning_trace:
            trace_loops = self._count_trace_loops(reasoning_trace)

        total_loops = activation_loops + trace_loops

        # Convergence check
        conv_trend = self._convergence_trend()
        is_converging = conv_trend < self.convergence_tolerance

        # φ-alignment: how close to golden FP
        phi_alignment = self._phi_alignment(flat)

        # Contact 3-form proxy
        contact_proxy = self._contact_proxy()

        # Classification
        closure_type = self._classify(total_loops, is_converging, phi_alignment)
        is_closed = closure_type == 'insight'
        is_circular = closure_type == 'circular'
        in_loop = total_loops >= self.CLOSURE_THRESHOLD

        # Track loop depth (how many consecutive detections of in_loop)
        loop_depth = self._loop_depth_from_history()

        self._total_detections += 1
        self._loop_type_counts[closure_type] += 1

        return {
            'loops': total_loops,
            'activation_loops': activation_loops,
            'trace_loops': trace_loops,
            'closure_type': closure_type,
            'is_closed': is_closed,
            'is_circular': is_circular,
            'in_closure_loop': in_loop,
            'loop_depth': loop_depth,
            'convergence_trend': float(conv_trend),
            'is_converging': is_converging,
            'phi_alignment': float(phi_alignment),
            'contact_3form_proxy': float(contact_proxy),
            'threshold': self.CLOSURE_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Activation loop detection
    # ------------------------------------------------------------------

    def _count_activation_loops(
        self,
        current_flat: np.ndarray,
        history: List[np.ndarray]
    ) -> int:
        """
        Count how many historical states are similar to the current state.
        Similarity = cosine similarity > threshold.
        """
        if not history:
            return 0

        curr_norm = np.linalg.norm(current_flat)
        if curr_norm < 1e-12:
            return 0
        curr_unit = current_flat / curr_norm

        loops = 0
        for prev in history[-self.window:]:
            prev_arr = np.asarray(prev, dtype=np.float64)
            if prev_arr.shape != current_flat.shape:
                # Resize if shapes differ
                if len(prev_arr) > len(current_flat):
                    prev_arr = prev_arr[:len(current_flat)]
                else:
                    tmp = np.zeros_like(current_flat)
                    tmp[:len(prev_arr)] = prev_arr
                    prev_arr = tmp

            prev_norm = np.linalg.norm(prev_arr)
            if prev_norm < 1e-12:
                continue

            sim = float(np.dot(curr_unit, prev_arr / prev_norm))
            if sim > self.similarity_threshold:
                loops += 1

        return loops

    # ------------------------------------------------------------------
    # Trace loop detection
    # ------------------------------------------------------------------

    def _count_trace_loops(self, current_trace: str) -> int:
        """
        Count repeated n-grams between current trace and recent history.
        A 'loop' in reasoning = the same 4+ word sequence appears in both
        the current trace and a recent previous trace.
        """
        if not self._trace_history:
            return 0

        current_ngrams = self._extract_ngrams(current_trace, n=4)
        if not current_ngrams:
            return 0

        loops = 0
        for prev_trace in list(self._trace_history)[:-1]:
            prev_ngrams = self._extract_ngrams(prev_trace, n=4)
            overlap = len(current_ngrams & prev_ngrams)
            if overlap >= 2:  # ≥2 matching 4-grams = likely circular reasoning
                loops += 1

        return loops

    @staticmethod
    def _extract_ngrams(text: str, n: int = 4) -> frozenset:
        """Extract all n-grams from text as a frozenset."""
        words = text.lower().split()
        if len(words) < n:
            return frozenset()
        ngrams = frozenset(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
        return ngrams

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _convergence_trend(self) -> float:
        """
        Compute step-to-step L2 distance trend over convergence_window steps.
        Negative = getting closer (converging).
        Positive = getting further (diverging).
        Near-zero = stable orbit (potential circular).
        """
        hist = list(self._state_history)
        if len(hist) < self.convergence_window + 1:
            return float('inf')  # not enough history

        recent = hist[-self.convergence_window - 1:]
        dists = [
            float(np.linalg.norm(
                np.asarray(recent[i+1], dtype=np.float64) -
                np.asarray(recent[i], dtype=np.float64)
            ))
            for i in range(len(recent) - 1)
        ]

        if len(dists) < 2:
            return float(np.mean(dists)) if dists else float('inf')

        # Trend: difference between last half and first half average distances
        mid = len(dists) // 2
        early_mean = np.mean(dists[:mid])
        late_mean = np.mean(dists[mid:])
        return float(late_mean - early_mean)  # negative = converging

    def _loop_depth_from_history(self) -> int:
        """
        Estimate how many consecutive time-steps have had closure loops.
        Approximated from the state history similarity structure.
        """
        hist = list(self._state_history)
        if len(hist) < 2:
            return 0

        depth = 0
        for i in range(len(hist) - 1, 0, -1):
            curr = np.asarray(hist[i], dtype=np.float64)
            prev = np.asarray(hist[i-1], dtype=np.float64)
            n_c = np.linalg.norm(curr)
            n_p = np.linalg.norm(prev)
            if n_c < 1e-12 or n_p < 1e-12:
                break
            sim = float(np.dot(curr / n_c, prev / n_p))
            if sim > self.similarity_threshold:
                depth += 1
            else:
                break

        return depth

    # ------------------------------------------------------------------
    # Fixed-point observables
    # ------------------------------------------------------------------

    def _phi_alignment(self, state: np.ndarray) -> float:
        """
        How close is the state's mean magnitude to φ⁻¹?
        1.0 = at golden FP mean. 0.0 = fully diverged.
        """
        state_mean = float(np.mean(np.abs(state)))
        alignment = 1.0 - abs(state_mean - GOLDEN_FP) / max(GOLDEN_FP, state_mean + 1e-12)
        return float(np.clip(alignment, 0.0, 1.0))

    def _contact_proxy(self) -> float:
        """
        Proxy for the contact 3-form density from the state history.
        Uses the last 3 states as L, R, C surrogates.
        Proxy α∧dα = mean(|L| + |R| + |C|) / 3.
        Target: 3φ⁻¹ ≈ 1.854 (at golden FP).
        """
        hist = list(self._state_history)
        if len(hist) < 3:
            return 3.0 * GOLDEN_FP  # default = target

        L = np.asarray(hist[-3], dtype=np.float64)
        R = np.asarray(hist[-2], dtype=np.float64)
        C = np.asarray(hist[-1], dtype=np.float64)

        return float(np.mean(np.abs(L) + np.abs(R) + np.abs(C)))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _classify(
        self,
        loops: int,
        is_converging: bool,
        phi_alignment: float
    ) -> ClosureType:
        """
        Classify the closure type based on loop count, convergence, and FP alignment.

        insight:    ≥3 loops + converging + high φ-alignment
        productive: ≥3 loops + converging OR moderate φ-alignment
        circular:   ≥3 loops + NOT converging + low φ-alignment
        open:       <3 loops
        """
        if loops < self.CLOSURE_THRESHOLD:
            return 'open'

        # All conditions met: genuine insight
        if is_converging and phi_alignment > 0.7:
            return 'insight'

        # Converging but not yet at FP: productive loop
        if is_converging or phi_alignment > 0.4:
            return 'productive'

        # Not converging, not near FP: degenerate circular loop
        return 'circular'

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear state history. Use after major architectural changes."""
        self._state_history.clear()
        self._trace_history.clear()

    def summary(self) -> Dict[str, Any]:
        return {
            'similarity_threshold': self.similarity_threshold,
            'window': self.window,
            'history_len': len(self._state_history),
            'total_detections': self._total_detections,
            'loop_type_counts': dict(self._loop_type_counts),
            'closure_threshold': self.CLOSURE_THRESHOLD,
        }


if __name__ == "__main__":
    detector = SelfReferentialClosureDetector(similarity_threshold=0.90, window=15)

    # Simulate convergence toward golden FP
    print("=== Convergence scenario (→ insight) ===")
    state = np.random.randn(64) * 2.0
    for step in range(25):
        state = state * 0.8 + np.full(64, GOLDEN_FP) * 0.2  # converge toward FP
        state += np.random.randn(64) * 0.01
        result = detector.detect(state, mode='activation')
        if result['loops'] > 0 or step > 18:
            print(f"  Step {step:2d}: loops={result['loops']}, "
                  f"type={result['closure_type']}, "
                  f"conv_trend={result['convergence_trend']:+.4f}, "
                  f"φ_align={result['phi_alignment']:.3f}")

    detector.reset_history()

    # Simulate oscillation (circular)
    print("\n=== Oscillation scenario (→ circular) ===")
    base = np.random.randn(64)
    for step in range(25):
        # Oscillate between two states
        state = base if step % 2 == 0 else -base
        state += np.random.randn(64) * 0.02
        result = detector.detect(state, mode='activation')
        if result['loops'] > 0 or step > 18:
            print(f"  Step {step:2d}: loops={result['loops']}, "
                  f"type={result['closure_type']}, "
                  f"conv_trend={result['convergence_trend']:+.4f}")

    print("\n=== Summary ===")
    print(detector.summary())
