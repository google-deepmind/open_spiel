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
* still experimental, but soon to be stable `flax.nnx` which much closer to the OOP paradigm


### Changelog:
1. Fully rewritten `tensorflow` model to `jax`, supported in 2 APIs
2. Rewritten utils like replay buffer and configuation classes to support the device-agnostic implementation


### TODOs:

1. Add complete tests for both APIs and conduct study that benchmarks the implementations on a selection of games
2. Fix multiprocessing/vectorisation overlap, enhancing the performance.



