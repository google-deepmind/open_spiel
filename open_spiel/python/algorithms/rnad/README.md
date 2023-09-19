This folder contains an single process implementation of [R-NaD]
(https://arxiv.org/pdf/2206.15378.pdf)

- `rnad.py` contains a reference implementation of the actor behavior and the
policy and value loss used in to train DeepNash. It uses much smaller network
architecture (an MLP) and is only able to run on smaller games.

- `rnad_nashconv_leduc.png` shows the evolution of the NashConv metric (a
distance to the Nash equilibrium) as the learning progress.

To generate these plots we used the following parameters:

| Hyper-parameter | Value |
| ----------- | ----------- |
| policy_network_layers | (256, 256) |
| eta_reward_transform | 0.2 |
| learning_rate | 5e-5 |
| clip_gradient | 10e4 |
| beta_neurd | 2.0 |
| clip_neurd | 10e4 |
| b1_adam | 0.0 |
| b2_adam | 0.999 |
| epsilon_adam | 10e-8 |
| target_network_avg | 10e-3 |
| rho_vtrace | np.inf |
| c_vtrace | 1.0 |
| trajectory_max | 10 |
| batch_size | 512 |
| entropy_schedule_size | (50000,) |
| entropy_schedule_repeats | (1,)|
| state_representation | "info_set" |
| policy_option.threshold | 0.03 |
| policy_option.discretization | 32 |
| finetune_from | -1 |

Finally, the seed used were in [0, 1, 2, 3, 4] and the learning lasted for at
most than 7M steps.
