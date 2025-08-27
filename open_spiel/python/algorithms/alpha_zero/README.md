## Python AlphaZero

This is a pure python implementation of the AlphaZero algorithm.For more information, please take a look at the
[full documentation](https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md). 

This is a pure python implementation of the AlphaZero algorithm. It's based on `flax` library for neural networks in `jax` and provides.

The code is arranged in the following way:

```Bash
.
├── alpha_zero.py
├── analysis.py
├── evaluator_test.py
├── evaluator.py
├── export_model.py
├── model_jax.py
├── model_nnx.py #not yet fine
├── model_test.py
├── model.py
```

Each file implements the following parts of the main documentation:
* [model](model.py)
* [export_model](export_model.py)
* [MCTS evaluator](evaluator.py)
* [analysis script](analysis.py)
* [the main script](alpha_zero.py)


## Note of `flax` APIs

Currently, the framework supports two APIs:
* currently stable `flax.linen` that encompasses functional paradigm
* still experimental, but soon to be stable `flax.nnx` which much closer to the OOP paradigm. We mostly focus on the refactoring of the existing solution, but there're some additional opportunities provided by the `flax.nnx` lifted tranforms: [examples](https://github.com/google/flax/blob/main/examples/nnx_toy_examples/)


### Changelog:
1. Fully rewritten `tensorflow` model to `jax`, supported in 2 APIs, that could be used interchangeably
2. Rewritten utils like replay buffer and configuation classes to support the device-agnostic implementation
3. Added `vmap` to the modules for the batched computation
4. Added full test coverage for the utility
5. [TODO] Added a Tensorboard support for the loggning

## Challenges (contributions are open!)
1. Implement sharding for multi-processing or multi-hostage (`xmap, jax.shard_map`) training and inference
2. Compile learning process (`model.update`) using a parallel associative scan (`jax.lax.scan`)
3. Add support of different logging methods, like `wandb` and such




