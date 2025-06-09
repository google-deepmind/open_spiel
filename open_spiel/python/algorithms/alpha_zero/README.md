## Python AlphaZero

This is a pure python implementation of the AlphaZero algorithm.
<!-- 
Note: this version is based on Tensorflow V1 and is no longer maintained.

For more information, please take a look at the
[full documentation](https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md). -->

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
├── model_nnx.py
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


