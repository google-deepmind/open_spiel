import argparse
import os
import random
import docker

experiment_params = {
    'naive_learner_ipd': {
        'game': 'ipd',
        'correction_type': 'none',
        'discount': 0.96,
        'policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 4096,
        'game_iterations': 150,
        'epochs': 200
    },
    'lola_ipd': {
        'game': 'ipd',
        'correction_type': 'lola',
        'use_opponent_modelling': False,
        'discount': 0.96,
        'policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 4096,
        'game_iterations': 150,
        'epochs': 200
    },
    'lola_ipd_om': {
        'game': 'ipd',
        'correction_type': 'lola',
        'use_opponent_modelling': False,
        'discount': 0.96,
        'policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 4096,
        'game_iterations': 150,
        'epochs': 200
    },
    'dice_ipd_one_step': {
        'game': 'ipd',
        'correction_type': 'dice',
        'use_opponent_modelling': False,
        'discount': 0.96,
        'policy_lr': 0.1,
        'opp_policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 1024,
        'game_iterations': 150,
        'epochs': 200,
        'critic_mini_batches': 1,
        'n_lookaheads': 1,
    },
    'dice_ipd_two_step': {
        'game': 'ipd',
        'correction_type': 'dice',
        'use_opponent_modelling': False,
        'discount': 0.96,
        'policy_lr': 0.1,
        'opp_policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 1024,
        'game_iterations': 150,
        'epochs': 200,
        'critic_mini_batches': 1,
        'n_lookaheads': 2,
    },
    'dice_ipd_three_step': {
        'game': 'ipd',
        'correction_type': 'dice',
        'use_opponent_modelling': False,
        'discount': 0.96,
        'policy_lr': 0.1,
        'opp_policy_lr': 0.1,
        'critic_lr': 0.3,
        'batch_size': 1024,
        'game_iterations': 150,
        'epochs': 200,
        'critic_mini_batches': 1,
        'n_lookaheads': 3,
    },
    'dice_ipd_two_step_om': {
        'game': 'ipd',
        'correction_type': 'dice',
        'use_opponent_modelling': True,
        'opp_policy_mini_batches': 8,
        'opponent_model_learning_rate': 0.125,
        'discount': 0.96,
        'policy_lr': 0.2,
        'opp_policy_lr': 0.3,
        'critic_lr': 0.1,
        'batch_size': 1024,
        'game_iterations': 150,
        'epochs': 200,
        'critic_mini_batches': 1,
        'n_lookaheads': 2,
    }
}


def main(args):
    with open('.env', 'r') as f:
        env_file = list(f.readlines())
    params = experiment_params[args.exp_name]
    params['exp_name'] = args.exp_name
    client = docker.from_env()
    random.seed(args.seed)
    seeds = random.sample(range(1000, 10000), args.num_seeds)
    print(f'Running experiment "{args.exp_name}" with seeds: {seeds}')
    print('Experiment parameters:')
    [print(f'  {k}={v}') for k, v in sorted(params.items())]
    for seed in seeds:
        params['seed'] = seed
        client.containers.run(
            image="open_spiel/lola:latest",
            command=' '.join([f'--{k}={v}' for k, v in params.items()]),
            name=f'{args.exp_name}_{seed}',
            detach=True,
            #user=f'{os.getuid()}:{os.getgid()}',
            #volumes={os.getcwd(): {'bind': '/open_spiel', 'mode': 'rw'}},
            device_requests=[
                docker.types.DeviceRequest(device_ids=["all"], capabilities=[['gpu']])
            ],
            environment=env_file
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args()
    main(args)
