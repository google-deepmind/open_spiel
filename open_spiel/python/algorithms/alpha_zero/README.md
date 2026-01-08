## Python AlphaZero

This is a pure python implementation of the AlphaZero algorithm.For more information, please take a look at the
[full documentation](https://github.com/deepmind/open_spiel/blob/master/docs/alpha_zero.md). 

This is a pure python implementation of the AlphaZero algorithm. It's based on `flax` library for neural networks in `jax`.

The code is arranged in the following way:

```Bash
.
├── alpha_zero.py # main script
├── analysis.py # experiments' results plotting
├── evaluator.py # mcts evaluator
├── export_model.py # model conversion util
├── model_linen.py # AZ in flax.linen
├── model_nnx.py # AZ in flax.nnx
├── replay_buffer.py # simple temporary replay buffer
└── utils.py
```

> [!NOTE]
> Before running the code, you might want to install additional [requirements](../../../scripts/python_extra_deps.sh).
> `jax` has to be re-installed to match the available hardware, see: [jax documentation](https://docs.jax.dev/en/latest/installation.html)

Each file implements the following parts of the main documentation:
* [model (with linen)](model_linen.py) or [model (with nnx)](model_nnx.py) 
* [export_model](export_model.py), to be able save or initialise the model
* [MCTS evaluator](evaluator.py), to run evaluation
* [analysis script](analysis.py), to plot the results of the experiment in the visual form
* [the main script](alpha_zero.py)

## Example Usage

The example script is located in [examples folder](../../examples/alpha_zero.py)

```Bash
python alpha_zero.py --path absolute/path/to/checkpoint/dir
```

To visualise the results, using the [analysis.py](./analysis.py):
```Bash
python analysis.py --path absolute/path/to/checkpoint/dir
```

Example script for `connect-four`:

```Bash
cd ./open_spiel/python/examples
python alpha_zero.py --game "connect_four" \  
                    --uct_c 2 \
                    --max_simulations 100 \
                    --train_batch_size 1024 \
                    --replay_buffer_size 2**14 \
                    --replay_buffer_reuse 4 \
                    --learning_rate 0.001 \
                    --weight_decay 0.0001 \
                    --temperature_drop 10 \
                    --nn_model "resnet" \
                    --nn_depth 10 \
                    --path absolute/path/to/chkpt \
                    --evaluation_window 100 \
                    --actors 2 \
                    --evaluators 1
```

## Notes about the implementation

Currently, the framework supports two APIs:
* currently stable `flax.linen` that encompasses functional paradigm
* still experimental, but soon to be stable `flax.nnx` which much closer to the OOP paradigm. We mostly focus on the refactoring of the existing solution, but there're some additional opportunities provided by the `flax.nnx` lifted tranforms: [examples](https://github.com/google/flax/blob/main/examples/nnx_toy_examples/)

## Changelog:
1. Fully rewritten `tensorflow` model to `jax`, supported in 2 APIs, that could be used interchangeably
2. Rewritten utils like replay buffer and configuation classes to support the device-agnostic implementation
3. Added `vmap` to the modules for the batched computation
4. Added full test coverage for the utility


## Challenges (contributions are open!)
1. Implement sharding for multi-processing or multi-hostage (`xmap, jax.shard_map`) training and inference to reduce the overhead.
2. Add support of different logging methods, like `wandb` and such
3. Add Tensorboard support for a "on-line" logging



