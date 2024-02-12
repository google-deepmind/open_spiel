import os
# 
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6 Mainly controlles the number of spawned threateds 
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

import argparse
from distutils.util import strtobool
import time
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from open_spiel.python.mfg import utils
from open_spiel.python import rl_environment
from open_spiel.python import policy as policy_std
from open_spiel.python.mfg.algorithms import distribution
from open_spiel.python.mfg.algorithms.nash_conv import NashConv
from open_spiel.python.mfg.algorithms import policy_value
from open_spiel.python.mfg.games import factory
from open_spiel.python.mfg import value
from open_spiel.python.mfg.algorithms import best_response_value




def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="set a random seed")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="Set the name of this experiment")
    parser.add_argument("--game-setting", type=str, default="crowd_modelling_2d_four_rooms", help="Set the game to benchmark options:(crowd_modelling_2d_four_rooms) and (crowd_modelling_2d_maze)")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of the optimizer")
    parser.add_argument("--num-episodes", type=int, default=5, help="set the number of episodes of to collect per rollout")
    parser.add_argument("--update-episodes", type=int, default=20, help="set the number of episodes of the inner loop")
    parser.add_argument("--update-iterations", type=int, default=100, help="Set the number of global update steps of the outer loop")
    
    parser.add_argument('--optimizer', type=str, default="Adam", help="Set the optimizer (Adam) or (SGD)")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="Use to repreduce experiment results")
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True, help="Use Gpu to run the experiment")

    
    # PPO parameters
    parser.add_argument('--gamma', type=float, default=0.9, help='set discount factor gamma')
    parser.add_argument("--num-minibatches", type=int, default=5,  help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=5, help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--max-grad-norm", type=float, default=5, help="the maximum norm for the gradient clipping")


    # MFPPO parameters
    parser.add_argument('--alpha', type= int, default=0.5, help='Set alpha to controll the iteration and epsiode policy updates')
    parser.add_argument('--eps-eps', type= int, default=0.2, help='eps to update the episode learned policy')
    parser.add_argument('--itr-eps', type= int, default=0.05, help='eps to update the episode learned policy')

    args = parser.parse_args()

    return args


class NashC(NashConv):
    # Mainly used to calculate the exploitability 
    def __init__(self, game,distrib,pi_value, root_state=None):
        self._game = game
        if root_state is None:
            self._root_states = game.new_initial_states()
        else:
            self._root_states = [root_state]
            
        self._distrib = distrib

        self._pi_value = pi_value

        self._br_value = best_response_value.BestResponse(
            self._game,
            self._distrib,
            value.TabularValueFunction(self._game),
            root_state=root_state)


class Agent(nn.Module):
    def __init__(self, info_state_size, num_actions):
        super(Agent, self).__init__()
        self.num_actions = num_actions
        self.info_state_size = info_state_size
        self.critic = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)), 
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,1))
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(info_state_size, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128,128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, num_actions))
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, bias_const=0.0):
    # used to initalize layers
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOpolicy(policy_std.Policy):
    # required obeject to work with openspiel 
    # used in updating the distribution using the policy 
    # and in calculating the nash-convergance 

    def __init__(self, game, agent, player_ids, device):
        super().__init__(game, player_ids)
        self.agent = agent
        self.device = device

    def action_probabilities(self, state, player_id=None):
        # main method that is called to update the population states distribution
        obs = torch.Tensor(state.observation_tensor()).to(self.device)
        legal_actions = state.legal_actions()
        logits = agent.actor(obs).detach().cpu()
        legat_logits = np.array([logits[action] for action in legal_actions])
        probs = np.exp(legat_logits -legat_logits.max())
        probs /= probs.sum(axis=0)
        
        # returns a dictionary with actions as keys and their probabilities as values
        return {action:probs[legal_actions.index(action)] for action in legal_actions}


def rollout(env, iter_agent, eps_agent, num_epsiodes, steps, device):
    # generates num_epsiodes rollouts
    info_state = torch.zeros((steps,iter_agent.info_state_size), device=device)
    actions = torch.zeros((steps,), device=device)
    logprobs = torch.zeros((steps,), device=device)
    rewards = torch.zeros((steps,), device=device)
    dones = torch.zeros((steps,), device=device)
    values = torch.zeros((steps,), device=device)
    entropies = torch.zeros((steps,), device=device)
    t_actions = torch.zeros((steps,), device=device)
    t_logprobs = torch.zeros((steps,), device=device)

    step = 0
    for _ in range(num_epsiodes):
        time_step = env.reset()
        while not time_step.last():
            obs = time_step.observations["info_state"][0]
            obs = torch.Tensor(obs).to(device)
            info_state[step] = obs
            with torch.no_grad():
                t_action, t_logprob, _, _ = iter_agent.get_action_and_value(obs)
                action, logprob, entropy, value = eps_agent.get_action_and_value(obs)

            time_step = env.step([action.item()])

            # iteration policy data
            t_logprobs[step] = t_logprob
            t_actions[step] = t_action

            # episode policy data
            logprobs[step] = logprob
            dones[step] = time_step.last()
            entropies[step] = entropy
            values[step] = value
            actions[step] = action
            rewards[step] = torch.Tensor(time_step.rewards).to(device)
            step += 1

    return info_state, actions, logprobs, rewards, dones, values, entropies,t_actions,t_logprobs 

def cal_Adv(gamma, norm, rewards,values, dones):
    # function used to calculate the Generalized Advantage estimate
    # using the exact method in stable-baseline3
    with torch.no_grad():
        next_done = dones[-1]
        next_value = values[-1] 
        steps = len(values)
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(steps)):
            if t == steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return

        advantages = returns - values

    if norm:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def update(update_epochs, num_minibatch, obs, logprobs, actions, advantages, returns, t_actions, t_logprobs, optimizer_actor, optimize_critic, agent, alpha = 0.5, t_eps = 0.2, eps = 0.2):
    # update the agent network (actor and critic)
    batch_size = actions.shape[0]
    b_inds = np.arange(batch_size)
    mini_batch_size = batch_size // num_minibatch
    # get batch indices
    np.random.shuffle(b_inds)
    for _ in range(update_epochs):
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = b_inds[start:end]
            # for each update epoch shuffle the batch indices
            # generate the new logprobs, entropy and value then calculate the ratio
            b_obs = obs[mb_inds]
            b_advantages = advantages[mb_inds]

            # Get the data under the episode policy (representative agent current policy)
            _, newlogprob, entropy, new_value = agent.get_action_and_value(b_obs, actions[mb_inds])
            logratio = newlogprob - logprobs[mb_inds]
            ratio = torch.exp(logratio)

            # Get the data under the iteration policy (the population policy)
            _, t_newlogprob, _, _ = agent.get_action_and_value(b_obs, t_actions[mb_inds])
            t_logratio = t_newlogprob - t_logprobs[mb_inds]
            t_ratio = torch.exp(t_logratio)

            # iteration update PPO
            t_pg_loss1 = b_advantages * t_ratio
            t_pg_loss2 = b_advantages * torch.clamp(t_ratio, 1 - t_eps, 1 + t_eps)
            
            # episodic update PPO 
            pg_loss1 = b_advantages * ratio
            pg_loss2 = b_advantages * torch.clamp(ratio, 1 - eps, 1 + eps)

            # Calculate the loss using our loss function 
            pg_loss = - alpha * torch.min(pg_loss1, pg_loss2).mean() - (1-alpha) * torch.min(t_pg_loss1, t_pg_loss2).mean()
            v_loss = F.smooth_l1_loss(new_value.reshape(-1), returns[mb_inds]).mean()
            entropy_loss = entropy.mean()

            loss = pg_loss - args.ent_coef * entropy_loss 
            
            # Actor update 
            optimizer_actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
            optimizer_actor.step()
            
            # Critic update 
            optimize_critic.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
            optimize_critic.step()

    return v_loss

def plot_dist(env, game_name, distrib, info_state, save=False, filename="agent_dist.mp4"):
    # this functions is used to generate an animated video of the distribuiton propagating throught the game 
    horizon = env.game.get_parameters()['horizon']
    size = env.game.get_parameters()['size']
    if game_name == "maze":
        d_size = 21
    else:
        d_size = 13
    agent_dist = np.zeros((horizon,d_size,d_size))
    mu_dist = np.zeros((horizon,d_size,d_size))


    for k,v in distrib.distribution.items():
        if "mu" in k:
            tt = k.split("_")[0].split(",")
            x = int(tt[0].split("(")[-1])
            y = int(tt[1].split()[-1])
            t = int(tt[2].split()[-1].split(")")[0])
            mu_dist[t,y,x] = v

    for i in range(horizon):
        obs = info_state[i].tolist()
        obs_x = obs[:size].index(1)
        obs_y = obs[size:2*size].index(1)
        obs_t = obs[2*size:].index(1)
        agent_dist[obs_t,obs_y,obs_x] = 0.02

    final_dist = agent_dist + mu_dist

    if save:
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(img, animated=True)] for img in final_dist]
        ani = animation.ArtistAnimation(fig, ims, blit=True, interval = 200)

        ani.save(filename, fps=5)

        plt.close()

def log_metrics(it,distrib, policy, writer, reward, entropy):
    # this function is used to log the results to tensor board
    initial_states = game.new_initial_states()
    pi_value = policy_value.PolicyValue(game, distrib, policy, value.TabularValueFunction(game))
    m = {
        f"ppo_br/{state}": pi_value.eval_state(state)
        for state in initial_states
    }
    m["nash_conv_ppo"] = NashC(game, distrib, pi_value).nash_conv()
    writer.add_scalar("initial_state_value", m['ppo_br/initial'], it)
    # debug
    writer.add_scalar("rewards", reward, it)
    writer.add_scalar("entorpy", entropy, it)

    writer.add_scalar("nash_conv_ppo", m['nash_conv_ppo'], it)
    logger.debug(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
    print(f"ppo_br: {m['ppo_br/initial']}, and nash_conv: {m['nash_conv_ppo']}, reward: {reward}, entropy: {entropy}")
    return m["nash_conv_ppo"]
    

if __name__ == "__main__":
    args = parse_args()

    # Set the seed 
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    # choose a value for the best model 
    # lower than which we save the weights and distribution 
    best_model = 300
    
    # Set the device (in our experiments CPU vs GPU does not improve time at all) we recommend CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Set the file name
    fname = "New_exp/maze_all_exp"
    
    # logging 
    run_name = f"{args.exp_name}_{args.game_setting}_{args.optimizer}_num_update_epochs_{args.update_epochs}_num_episodes_per_rollout_{args.num_episodes}_number_of_mini_batches_{args.num_minibatches}_{time.asctime(time.localtime(time.time()))}"
    log_name = os.path.join(fname, run_name)
    tb_writer = SummaryWriter(log_name)
    LOG = log_name + "_log.txt"                                                    
    logging.basicConfig(filename=LOG, filemode="a", level=logging.DEBUG, force=True)  

    # console handler  
    console = logging.StreamHandler()  
    console.setLevel(logging.ERROR)  
    logging.getLogger("").addHandler(console)
    
    logger = logging.getLogger()
    logger.debug("Initialization")

    tb_writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key,value in vars(args).items()])),
    )
    
    # Create the game instance 
    game = factory.create_game_with_setting("mfg_crowd_modelling_2d", args.game_setting)

    # Set the initial policy to uniform and generate the distribution 
    uniform_policy = policy_std.UniformRandomPolicy(game)
    mfg_dist = distribution.DistributionPolicy(game, uniform_policy)
    env = rl_environment.Environment(game, mfg_distribution=mfg_dist, mfg_population=0)

    # Set the environment seed for reproduciblility 
    env.seed(args.seed)

    # Creat the agent and population policies 
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]
    agent = Agent(info_state_size,num_actions).to(device)
    ppo_policy = PPOpolicy(game, agent, None, device)
    pop_agent = Agent(info_state_size,num_actions).to(device)

    if args.optimizer == "Adam":
        optimizer_actor = optim.Adam(agent.actor.parameters(), lr=args.lr,eps=1e-5)
        optimizer_critic = optim.Adam(agent.critic.parameters(), lr=args.lr,eps=1e-5)
    else:
        optimizer_actor = optim.SGD(agent.actor.parameters(), lr=args.lr, momentum=0.9)
        optimizer_critic = optim.SGD(agent.critic.parameters(), lr=args.lr, momentum=0.9)

    # Used to log data for debugging
    steps = args.num_episodes * env.max_game_length
    episode_entropy = []
    total_entropy = []
    Nash_con_vect = []

    eps_reward = []
    total_reward = []

    for k in range(args.update_iterations):
        for eps in range(args.update_episodes):
            # collect rollout data
            obs, actions, logprobs, rewards, dones, values, entropies, t_actions, t_logprobs = rollout(env, pop_agent, agent, args.num_episodes, steps, device)
            #store rewards and entropy for debugging
            episode_entropy.append(entropies.mean().item())
            eps_reward.append(rewards.sum().item()/args.num_episodes)
            # Calculate the advantage function 
            adv, returns = cal_Adv(args.gamma, True, rewards, values, dones)
            # Update the learned policy and report loss for debugging
            v_loss = update(args.update_epochs,args.num_minibatches, obs, logprobs, actions, adv, returns, t_actions, t_logprobs, optimizer_actor, optimizer_critic, agent, args.alpha,args.itr_eps ,args.eps_eps)    
            
        #collect and print the metrics
        total_reward.append(np.mean(eps_reward))
        total_entropy.append(np.mean(episode_entropy))

        print("Value_loss", v_loss.item())
        print("iteration num:", k)
        print('Mean reward', total_reward[-1])    
        
        # Update the iteration policy with the new policy 
        pop_agent.load_state_dict(agent.state_dict())
        
        # Update the distribution 
        distrib = distribution.DistributionPolicy(game, ppo_policy)
        
        # calculate the exploitability 
        Nash_con_vect.append(log_metrics(k+1, distrib, ppo_policy, tb_writer, total_reward[-1], total_entropy[-1]))

        # update the environment distribution 
        env.update_mfg_distribution(distrib)
        

    if best_model >= Nash_con_vect[-1]:    
        #save the distribution and weights for further analysis 
        filename = os.path.join(fname, f"distribution_{run_name}.pkl")
        utils.save_parametric_distribution(distrib, filename)   
        torch.save(agent.actor.state_dict(),fname + f"alpha_{args.alpha}, itr_eps_{args.itr_eps}, eps_eps_{args.eps_eps}_agent_actor_weights.pth")
        torch.save(agent.critic.state_dict(),fname + f"alpha_{args.alpha}, itr_eps_{args.itr_eps}, eps_eps_{args.eps_eps}_agent_critic_weights.pth")
