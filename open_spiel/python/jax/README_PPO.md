# PPO Self-Play Agent (JAX)

A JAX implementation of Proximal Policy Optimization (PPO) for OpenSpiel games
with self-play support.

## Files

| File | Description |
|------|-------------|
| `open_spiel/python/jax/ppo.py` | PPO agent, actor-critic network, rollout buffer |
| `open_spiel/python/jax/ppo_utils.py` | Vectorized GAE, metrics tracking, plotting |
| `open_spiel/python/examples/ppo_example_jax.py` | Training script (single game) |
| `open_spiel/python/examples/ppo_benchmark_jax.py` | Multi-game benchmark with plots |
| `open_spiel/python/jax/ppo_jax_test.py` | Unit tests |

## Quick Start

```bash
# Train on Kuhn Poker
python -m open_spiel.python.examples.ppo_example_jax --game=kuhn_poker

# Train with higher entropy (better for mixed strategies)
python -m open_spiel.python.examples.ppo_example_jax \
    --game=kuhn_poker --entropy_coef=0.1 --num_iterations=500

# Train on a simultaneous game
python -m open_spiel.python.examples.ppo_example_jax --game=matrix_pd

# Run benchmarks across all games
python -m open_spiel.python.examples.ppo_benchmark_jax

# Run unit tests
python -m open_spiel.python.jax.ppo_jax_test
```

## Algorithm

### PPO (Proximal Policy Optimization)

PPO (Schulman et al., 2017) is an on-policy actor-critic algorithm that
optimizes a clipped surrogate objective to prevent destructively large policy
updates:

```
L_clip = E[ min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A) ]
```

where `r(theta) = pi_new(a|s) / pi_old(a|s)` is the probability ratio and `A`
is the advantage estimate. The total loss combines:

- **Policy loss**: clipped surrogate objective
- **Value loss**: MSE between predicted values and returns
- **Entropy bonus**: encourages exploration (critical for mixed strategies)

### GAE (Generalized Advantage Estimation)

GAE (Schulman et al., 2016) computes advantages by blending TD residuals with an
exponentially decaying weight:

```
delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
A_t     = delta_t + (gamma * lambda) * (1 - done_t) * A_{t+1}
```

#### Vectorized Implementation via jax.lax.scan

The naive implementation uses a Python for-loop in reverse time order. This
module replaces it with `jax.lax.scan(reverse=True)`:

```python
def _scan_step(last_gae, x):
    reward, value, next_value, done = x
    non_terminal = 1.0 - done
    delta = reward + gamma * next_value * non_terminal - value
    new_gae = delta + gamma * gae_lambda * non_terminal * last_gae
    return new_gae, new_gae

_, advantages = jax.lax.scan(
    _scan_step, jnp.float32(0.0),
    (rewards, values, next_values, dones),
    reverse=True)
```

Benefits:
- **No Python loops**: the scan body compiles to a single XLA while-loop
- **JIT-compatible**: can be wrapped in `jax.jit` for tracing and compilation
- **Functional**: pure function with no side effects or mutations

## JAX Design Choices

### PRNG Key Threading

All randomness uses `jax.random` with explicit key management. The agent
maintains a PRNG state `self._rng` and splits it for each random operation:

```python
def _next_rng(self) -> chex.PRNGKey:
    self._rng, subkey = jax.random.split(self._rng)
    return subkey

# Action sampling
action = jax.random.categorical(self._next_rng(), masked_logits)

# Minibatch shuffling
perm = jax.random.permutation(self._next_rng(), batch_size)
```

This ensures:
- Full reproducibility given the same seed
- No dependency on `numpy.random` global state
- Compatibility with JAX transformations (jit, vmap, etc.)

### JIT Compilation

The PPO loss and gradient update are JIT-compiled via the Flax NNX
`graphdef`/`merge` pattern:

```python
@jax.jit
def update(state, obs, actions, ...):
    network, optimizer = nn.merge(graphdef, state, copy=True)
    (loss, aux), grads = nn.value_and_grad(loss_fn, has_aux=True)(network, ...)
    optimizer.update(network, grads)
    return loss, aux, nn.state((network, optimizer))
```

Hyperparameters (clip_coef, value_coef, entropy_coef) are captured as closure
constants so they don't cause recompilation.

### JAX Arrays Everywhere

All data paths use `jnp` arrays:
- Observations and legal masks are converted to `jnp` in `step()`
- The rollout buffer converts to `jnp` via `as_jnp()` before training
- GAE computation is pure `jnp` + `jax.lax.scan`
- Minibatch indexing uses `jax.random.permutation`

## Self-Play

A single PPO agent instance controls all players. For sequential games, the
agent reads `time_step.current_player()` to determine which player's observation
to use. For simultaneous games, `step()` accepts a `player_id` parameter.

Per-player trajectories are tracked separately in `_episode_data`. At episode
boundaries, GAE is computed independently for each player's trajectory, then all
transitions are added to a shared rollout buffer for the PPO update.

## Benchmark Results

Tested on: `kuhn_poker`, `leduc_poker`, `matrix_pd`

### Kuhn Poker (best: `--entropy_coef=0.1`)

| Metric | Value |
|--------|-------|
| Best exploitability | ~0.22 |
| Final entropy | ~0.14 |
| Game value (P0) | ~-0.056 (theory: -1/18) |

### Expected Behavior

- **Exploitability** does not converge to zero. PPO is a single-policy gradient
  method and lacks convergence guarantees in imperfect-information games. CFR
  reaches near-zero exploitability in seconds for these games.
- **Entropy** tends to collapse without sufficient `entropy_coef`. Higher values
  (0.05-0.1) help maintain the mixed strategies required for Nash equilibrium.
- **Average returns** for both players should be near the game value, confirming
  the self-play mechanics work correctly.

### Generating Plots

```bash
python -m open_spiel.python.examples.ppo_benchmark_jax \
    --output_dir=/tmp/ppo_benchmark --num_iterations=500

# Individual game with plot
python -m open_spiel.python.examples.ppo_example_jax \
    --game=kuhn_poker --plot_path=kuhn_training.png
```

## Dependencies

Same as existing JAX agents in OpenSpiel -- no new packages:

- `jax`, `jax.numpy`
- `flax.nnx` (Flax NNX API)
- `optax` (optimizer, gradient clipping)
- `chex` (type checking)
- `matplotlib` (optional, for plotting only)
