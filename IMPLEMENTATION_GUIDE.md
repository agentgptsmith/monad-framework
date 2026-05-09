# Implementation Guide for Sonnet

## Recommended Coding Order

### Phase 0: Foundations (Do this first)
1. Set up basic project structure (folders: memory, generation, evolution, utils)
2. Implement core constants (φ, π, derived values)
3. Build simple vector/tensor helpers

### Phase 1: Memory Layer (Critical Foundation)
1. Implement φ-scaled Penrose-Beenker tiling
   - Start with 2D approximation if full tiling is heavy
   - Core function: distance-weighted bond strength `kappa = phi ** (-dist)`
2. Add basic store/retrieve with geometric decay
3. (Later) Add 3-torus topological protection layer

**Tip:** Make the tiling the single source of truth for conceptual/relational memory. Everything else reads/writes through it.

### Phase 2: Generation Layer (Feigenbaum Gap)
1. Implement δ_L and δ_S calculation functions
2. Build `sample_from_gap()` — this is the heart of structured mutation
3. Add geometric bias/constraint functions (favor φ-compatible changes)
4. Create simple mutation operator that uses the gap

**Tip:** Keep gap sampling tunable. Expose a "chaos_level" or "gap_width" parameter early.

### Phase 3: Evolutionary Layer (MOSES-style)
1. Basic program/tree representation
2. Integrate gap-based mutation
3. Add geometric fitness bonus (how well does this map to the tiling?)
4. Simple selection + replacement loop

**Tip:** Don't over-engineer the evolutionary algorithm at first. The gap mutation + geometric fitness will do most of the work.

### Phase 4: Topological Protection (3-Torus)
1. Implement basic 3-torus abstraction
2. Add non-contractible loop protection for critical state
3. Integrate with memory layer (optional high-reliability path)

### Phase 5: Cooling & Reset
1. Implement Topological Cooling Sequence (Samson feedback + symplectic collapse + LRC transaction)
2. Wire it after heavy generation phases

### Phase 6: Integration & Testing
1. Connect memory + generation + evolution
2. Build simple test cases (e.g., evolving structures that must remain compressible)
3. Add visualization for tiling, gap sampling, and cooling

## Key Tips

- **Geometry first**: Make φ-scaling and tiling distance first-class everywhere.
- **Gap is king**: `sample_from_gap()` should be clean, well-tested, and tunable.
- **Cooling is mandatory**: Never generate repeatedly without a reset path.
- **Start simple**: 2D tiling approximation is fine early on. Full aperiodic tiling can come later.
- **Document the "why"**: Especially around geometric fitness and gap mutation.

## Common Pitfalls to Avoid

- Treating mutation as pure randomness (use the gap)
- Forgetting to cool after generation
- Building complex evolutionary logic before the mutation + fitness foundation is solid
- Ignoring geometric constraints in fitness

---

*Practical guide. Hand this to Sonnet along with the architecture docs.*