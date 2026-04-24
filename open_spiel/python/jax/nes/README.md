# Neural Equilibrium solvers for game theory

## TODOs:

- [ ] Better logging utility, parallel LP solvers(?)
- [ ] Sharding and data/pipeline parallelisation for the architecture
- [ ] DID implementation


This section implements two papers covering efficient solving Normal-Form Games (NFG) with neural networks.

1. Turbocharging Solution Concepts: Solving NEs, CEs and CCEs with Neural Equilibrium Solvers (arXiv:2210.09257)
2. Deep Incentive Design with Differentiable Equilibrium Blocks

## NES (Neural Equilibrium solver)

Minimcal API:
```Python
solver = NESolver(
    game=pyspiel.load_game("matrix_rps"),  # or games.Game.L2_INVARIANT
    mode=networks.Mode.CCE,                # or Mode.CE
    network_config={
        "dual_channels": 32,
        "payoff_channel_list": [128, 64, 128],
        "dual_channel_list": [32, 32],
    },
    mu=1.0, rho=10.0,
    batch_size=32,
    network_train_steps=100,
)
result = solver.solve()  # → {"duals": α, "policy": σ}
```

### Motivation

This section implements a Neural Equilibrium Solver (NES) that learns to compute ε-approximate Correlated Equilibria (CE) and Correlated Course Equilibria (CCE) via dual optimization. The core idea is to train a neural network to predict dual variables α, from which primal variables (joint strategy σ and slack ε) are recovered analytically through the ε-MWMRE (Maximum Welfare Minimum Relative Entropy) objective.

## Method

The network consumes a 4-channel tensor of shape [B, 4, N, A1, ..., AN]:
Channel 0: Normalized payoffs `Ĝ`
Channel 1: Target epsilon `ε̂` (broadcasted)
Channel 2: Target joint strategy `σ̂` (broadcasted)
Channel 3: Welfare `W` (broadcasted)

The pipeline is following

```
Input: [B, 4, N, A1, ..., AN]
    ↓
[1] EquivariantPayoffToPayoff × K   (maintains joint action space)
    ↓
[2] PayoffsToDuals                  (marginalizes to player-action space)
    ↓
[3] EquivariantDualToDual × L       (refines in individual space)
    ↓
[4] Final projection + Softplus     (ensures α ≥ 0)
Output: α — [B, 1, N, A] for CCE, [B, 1, N, A, A] for CE
```

Given duals α, the closed-form primal recovery implements Equation (7) from the paper [1]:
1. Compute deviation contributions per player (`gain_p`)
2. Form logits: `l(a) = μ·W(a) − (1/ρ)·Σ_p gain_p(a)`
3. Recover `σ(a) ∝ σ̂(a) · exp(l(a))`
4. Recover `ε_p = (ε̂_p − ε⁺) · exp(−Σα_p / ρ) + ε⁺`


The dual loss minimized by `AdamW`:
```
L_dual = log_sum_exp + ε⁺ · Σ_p Σ_a α_p(a) − ρ · Σ_p ε_p
```

## DID (Deep Incentive Design with Deep Equilibrium blocks)

This section is currently under construcion.
