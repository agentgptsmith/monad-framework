"""
MONAD Framework — MOSES-Style Evolutionary Search
===================================================
MOSES = Meta-Optimizing Semantic Evolutionary Search, adapted here with:

1. Feigenbaum gap mutation (replaces standard Gaussian mutation)
2. Geometric fitness bonus (φ-tiling compressibility reward)
3. φ-harmonic initialization (population starts near golden FP)
4. Lucas-prime schedule checkpoints (evaluation at prime-indexed generations)
5. Phason-flip diversity injection (when population converges prematurely)

Standard MOSES problems with random mutation:
  - Uniform noise can destroy geometric structure
  - No preference for compressible solutions
  - No awareness of phase space topology

MONAD MOSES fixes these by sampling mutations from the Feigenbaum gap:
  - Mutations respect φ-scaling (geometrically coherent)
  - Geometric fitness bonus rewards compressibility in the Penrose tiling
  - Cooling prevents genetic drift away from the golden FP basin
  - Phason flips provide controlled diversity without destroying structure

Population representation:
  Programs are represented as real-valued parameter vectors of dimension `dim`.
  For actual program evolution, embed programs in this space (e.g., via
  tokenized representations, numeric parameters, or learned embeddings).

The (μ+λ) strategy:
  - μ parents selected from population
  - λ offspring generated via gap mutation of selected parents
  - Best μ from (parents + offspring) survive
  - Avoids generational loss while still applying selection pressure
"""

from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from core.constants import (
    PHI, PHI_INV, PHI_INV2, PHI_SQ, GOLDEN_FP,
    FEIGENBAUM_GAP, phi_harmonic_basis, LUCAS_PRIME_INDICES
)
from generation.state_dependent_gap import StateDependentFeigenbaumGap
from cooling.topological_cooling import TopologicalCoolingSequence


FitnessFunc = Callable[[np.ndarray], float]


class Individual:
    """A single individual in the population."""
    __slots__ = ('genome', 'fitness', 'geometric_bonus', 'total_fitness', 'age')

    def __init__(self, genome: np.ndarray):
        self.genome = genome.astype(np.float64)
        self.fitness = 0.0
        self.geometric_bonus = 0.0
        self.total_fitness = 0.0
        self.age = 0

    def __lt__(self, other: 'Individual') -> bool:
        return self.total_fitness < other.total_fitness


class MOSESEvolver:
    """
    MONAD-augmented MOSES evolutionary search.

    Usage:
        evolver = MOSESEvolver(dim=64, population_size=50)
        best = evolver.run(
            fitness_fn=my_fitness_function,
            n_generations=100
        )
    """

    def __init__(
        self,
        dim: int,
        population_size: int = 50,
        mu: int = 10,          # parents selected per generation
        lam: int = 20,         # offspring per generation
        geometric_weight: float = 0.1,
        cooling_every: int = 10,
        phason_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Args:
            dim: Dimensionality of the solution space.
            population_size: Total population size.
            mu: Number of parents selected each generation (mu+lambda strategy).
            lam: Number of offspring generated each generation.
            geometric_weight: Weight of geometric fitness bonus (0 = off, 0.1 = mild).
            cooling_every: Apply topological cooling to population every N generations.
            phason_rate: Probability of applying phason flip instead of phonon perturbation.
            seed: Random seed for reproducibility.
        """
        self.dim = dim
        self.pop_size = population_size
        self.mu = min(mu, population_size)
        self.lam = lam
        self.geometric_weight = geometric_weight
        self.cooling_every = cooling_every
        self.phason_rate = phason_rate

        if seed is not None:
            np.random.seed(seed)

        # Components
        self.gap = StateDependentFeigenbaumGap()
        self.cooling = TopologicalCoolingSequence(threshold=0.3, cooling_rate=PHI_INV)

        # Pre-compute φ-harmonic basis for geometric fitness
        self._phi_basis = phi_harmonic_basis(dim)

        # Population
        self.population: List[Individual] = []
        self._initialise_population()

        # History
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.phi_alignment_history: List[float] = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialise_population(self) -> None:
        """
        Initialize population near the golden fixed point.
        Small perturbations from φ⁻¹ ensure diversity while respecting the
        basin of attraction of the golden FP.
        """
        self.population = []
        for _ in range(self.pop_size):
            # Start at golden FP + small gap-width perturbation
            genome = np.full(self.dim, GOLDEN_FP, dtype=np.float64)
            genome += np.random.randn(self.dim) * FEIGENBAUM_GAP * 10
            self.population.append(Individual(genome))

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        fitness_fn: FitnessFunc,
        population: Optional[List[Individual]] = None
    ) -> None:
        """Evaluate fitness for all individuals, including geometric bonus."""
        pop = population if population is not None else self.population
        for ind in pop:
            # User-provided fitness
            try:
                ind.fitness = float(fitness_fn(ind.genome))
            except Exception:
                ind.fitness = 0.0

            # Geometric fitness bonus: φ-alignment reward
            ind.geometric_bonus = self.geometric_fitness(ind.genome)

            # Combined fitness
            ind.total_fitness = ind.fitness + self.geometric_weight * ind.geometric_bonus

    def geometric_fitness(self, genome: np.ndarray) -> float:
        """
        Geometric fitness bonus: reward φ-harmonic alignment.

        A genome that is more aligned with the φ-harmonic basis is more
        'compressible' in the Penrose-Beenker tiling — it maps more naturally
        to the geometric memory substrate.

        Returns value in [0, 1]: 0 = no alignment, 1 = perfectly φ-aligned.
        """
        genome_norm = np.linalg.norm(genome)
        if genome_norm < 1e-12:
            return 0.0
        alignment = abs(float(np.dot(genome / genome_norm, self._phi_basis)))
        return float(np.clip(alignment, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_parents(self) -> List[Individual]:
        """
        Tournament selection: select mu parents.
        Each tournament has size 3 (respects 3-loop threshold from contact geometry).
        """
        parents = []
        for _ in range(self.mu):
            tournament_idx = np.random.choice(len(self.population), size=3, replace=False)
            tournament = [self.population[i] for i in tournament_idx]
            winner = max(tournament, key=lambda ind: ind.total_fitness)
            parents.append(winner)
        return parents

    # ------------------------------------------------------------------
    # Mutation (gap-based)
    # ------------------------------------------------------------------

    def mutate(
        self,
        parent: Individual,
        architecture: str = 'rnn'
    ) -> Individual:
        """
        Create offspring via Feigenbaum gap mutation.

        With probability phason_rate: apply phason flip (discrete, larger)
        Otherwise: apply phonon perturbation (continuous, smaller)

        The phason flip breaks the parent out of a local attractor.
        The phonon perturbation explores locally.
        """
        genome = parent.genome.copy()

        if np.random.random() < self.phason_rate:
            # Phason flip: discrete rearrangement
            mutated = self.gap.phason_flip(genome)
        else:
            # Phonon perturbation: continuous gap-based nudge
            eff_gap = self.gap.compute_effective_gap(genome, architecture=architecture)
            pert = self.gap.sample_structured_perturbation(genome, eff_gap)
            mutated = genome + pert

        child = Individual(mutated)
        child.age = 0
        return child

    def crossover(self, parent_a: Individual, parent_b: Individual) -> Individual:
        """
        φ-weighted crossover: golden-ratio blend of two parents.
        Child = φ⁻¹ × parent_a + (1 - φ⁻¹) × parent_b
        This mirrors the Fibonacci sequence structure: each step blends with
        the golden ratio, converging toward the FP geometrically.
        """
        child_genome = PHI_INV * parent_a.genome + (1.0 - PHI_INV) * parent_b.genome
        return Individual(child_genome)

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------

    def step(
        self,
        fitness_fn: FitnessFunc,
        architecture: str = 'rnn',
        use_crossover: bool = False
    ) -> Individual:
        """
        Run one generation.
        Returns the current best individual.
        """
        # Select parents
        parents = self.select_parents()

        # Generate offspring
        offspring = []
        for i, parent in enumerate(parents):
            n_children = max(1, self.lam // self.mu)
            for _ in range(n_children):
                if use_crossover and len(parents) > 1:
                    partner = parents[(i + 1) % len(parents)]
                    child = self.crossover(parent, partner)
                    child = self.mutate(child, architecture)
                else:
                    child = self.mutate(parent, architecture)
                offspring.append(child)

        # Evaluate offspring
        self.evaluate_population(fitness_fn, offspring)

        # (μ+λ) selection: best from parents + offspring
        combined = self.population + offspring
        combined.sort(key=lambda ind: ind.total_fitness, reverse=True)
        self.population = combined[:self.pop_size]

        # Age existing individuals
        for ind in self.population:
            ind.age += 1

        # Apply cooling to the centroid if population has drifted
        if self.generation % self.cooling_every == 0:
            self._cool_population()

        # Track history
        best = self.population[0]
        mean_fit = float(np.mean([ind.total_fitness for ind in self.population]))
        mean_phi = float(np.mean([ind.geometric_bonus for ind in self.population]))
        self.best_fitness_history.append(best.total_fitness)
        self.mean_fitness_history.append(mean_fit)
        self.phi_alignment_history.append(mean_phi)

        self.generation += 1
        return best

    def run(
        self,
        fitness_fn: FitnessFunc,
        n_generations: int = 100,
        architecture: str = 'rnn',
        use_crossover: bool = False,
        verbose: bool = False,
        checkpoint_fn: Optional[Callable[[int, 'MOSESEvolver'], None]] = None
    ) -> Individual:
        """
        Run full evolutionary search.

        Checkpoints fire at Lucas-prime generation numbers (aligned with
        the prime-dark structure: prime-indexed generations are 'dark'
        — more informative, warranting closer inspection).

        Args:
            fitness_fn: User fitness function. Receives genome (np.ndarray), returns float.
            n_generations: Total generations to run.
            architecture: Architecture type for gap scaling.
            use_crossover: Enable φ-weighted crossover.
            verbose: Print progress.
            checkpoint_fn: Called at Lucas-prime generations with (gen, evolver).

        Returns:
            Best individual found.
        """
        # Initial evaluation
        self.evaluate_population(fitness_fn)
        self.population.sort(key=lambda ind: ind.total_fitness, reverse=True)

        lucas_checkpoints = set(StateDependentFeigenbaumGap.lucas_prime_schedule(n_generations))

        for gen in range(n_generations):
            best = self.step(fitness_fn, architecture, use_crossover)

            if gen in lucas_checkpoints:
                if checkpoint_fn:
                    checkpoint_fn(gen, self)
                if verbose:
                    print(f"  [L-prime gen {gen:4d}] best={best.total_fitness:.6f}  "
                          f"φ-align={best.geometric_bonus:.4f}  "
                          f"mean={self.mean_fitness_history[-1]:.6f}")

            elif verbose and gen % 10 == 0:
                print(f"  [gen {gen:4d}] best={best.total_fitness:.6f}  "
                      f"mean={self.mean_fitness_history[-1]:.6f}")

        return self.population[0]

    # ------------------------------------------------------------------
    # Population-level cooling
    # ------------------------------------------------------------------

    def _cool_population(self) -> None:
        """
        Apply cooling to the population centroid.
        If the centroid has drifted from the golden FP, cool all individuals
        proportionally toward the FP.
        """
        centroid = np.mean([ind.genome for ind in self.population], axis=0)
        if self.cooling.should_cool(centroid):
            cooled_centroid = self.cooling.apply(centroid)
            drift = cooled_centroid - centroid
            for ind in self.population:
                ind.genome += drift * PHI_INV  # gentle nudge, not full correction

    # ------------------------------------------------------------------
    # Lyapunov estimate (from fitness history)
    # ------------------------------------------------------------------

    def lyapunov_estimate(self, window: int = 10) -> float:
        """
        Estimate the local Lyapunov exponent from the best fitness history.
        Positive → chaotic (need cooling), negative → stable, near-zero → edge of chaos.
        This uses the same Lyapunov analysis as the Feigenbaum gap scaling.
        """
        if len(self.best_fitness_history) < window + 1:
            return 0.0

        recent = self.best_fitness_history[-window - 1:]
        diffs = np.diff(recent)
        log_diffs = np.log(np.abs(diffs) + 1e-12)
        return float(np.mean(log_diffs))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.population:
            return {}
        best = self.population[0]
        centroid = np.mean([ind.genome for ind in self.population], axis=0)
        return {
            'generation': self.generation,
            'pop_size': self.pop_size,
            'best_fitness': float(best.total_fitness),
            'best_geometric_bonus': float(best.geometric_bonus),
            'mean_fitness': float(np.mean([ind.total_fitness for ind in self.population])),
            'mean_phi_alignment': float(np.mean([ind.geometric_bonus for ind in self.population])),
            'centroid_mean': float(np.mean(np.abs(centroid))),
            'centroid_vs_fp': float(abs(np.mean(np.abs(centroid)) - GOLDEN_FP)),
            'lyapunov_estimate': self.lyapunov_estimate(),
            'cooling_urgency': float(self.cooling.cooling_urgency(centroid)),
            'global_coherence': float(self.cooling.global_coherence(centroid)),
        }


if __name__ == "__main__":
    # Demo: evolve toward the golden fixed point
    def sphere_fitness(genome: np.ndarray) -> float:
        """Maximize closeness to golden FP (negative sphere centered at φ⁻¹)."""
        return -float(np.mean((genome - GOLDEN_FP) ** 2))

    evolver = MOSESEvolver(
        dim=32,
        population_size=40,
        mu=10,
        lam=20,
        geometric_weight=0.2,
        seed=42
    )

    print("=== MONAD MOSES Evolver Demo ===")
    print(f"Initial best fitness: {evolver.population[0].total_fitness:.6f}")

    best = evolver.run(
        fitness_fn=sphere_fitness,
        n_generations=50,
        verbose=True
    )

    print(f"\nFinal summary:")
    for k, v in evolver.summary().items():
        print(f"  {k}: {v}")

    print(f"\nBest genome mean: {np.mean(best.genome):.6f}  (target: {GOLDEN_FP:.6f})")
    print(f"Best φ-alignment: {best.geometric_bonus:.6f}")
