# Feigenbaum Gap in Transformers

## Core Idea

The Dual-Class Feigenbaum gap can be used as a structured, state-dependent modulator of chaos and mixing inside Transformer architectures.

## Where to Apply the Gap

### 1. Attention Mixing
- Modulate attention temperature or entropy using gap width.
- Wider gap → higher effective chaos in token mixing → more creative / divergent attention patterns.
- Narrower gap → more focused, coherent attention.

### 2. Residual Stream Dynamics
- Add structured perturbations from the gap to the residual stream at each layer.
- This controls the effective Lyapunov exponent across depth.
- Useful for balancing coherence vs creativity in generation.

### 3. Key-Value Cache Management
- Use hierarchical Penrose-Beenker tiling to store KV cache entries with geometric relevance decay.
- Gap backwards can be used to reconstruct or compress older cache entries.

### 4. Layer-wise Curriculum
- Dynamically adjust gap width per layer during training or inference.
- Early layers: narrower gap (stable feature extraction)
- Middle layers: wider gap (rich mixing)
- Late layers: tunable gap (controlled creativity or focus)

## State-Dependent Formulation

```python
def compute_transformer_gap_width(layer_idx, attention_entropy, token_mixing_rate, base_gap):
    # Widen gap when attention is too focused or too diffuse
    mixing_factor = abs(attention_entropy - optimal_entropy) + token_mixing_rate
    return base_gap * (1.0 + mixing_factor)
```

## Benefits

- More principled control of output diversity than temperature scaling
- Architecture-aware chaos (gap responds to actual attention statistics)
- Natural integration with geometric memory (tiling for KV cache and important representations)
- Bidirectional capability (forward generation + backward reconstruction)

This turns the Transformer into a controllable chaotic dynamical system on the residual stream while keeping critical structures protected.

---

*Specific application of the Feigenbaum gap inside Transformer models.*