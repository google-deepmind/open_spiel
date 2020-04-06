"""
Test out the quiesce non_sparse and sparse with randomly generated matrixes. Compare it with gambit. If quiesce finds one of the gambit solutions, then regards it as a pass
"""
import numpy as np
import os
import sys
import time
import pdb
import pickle
from open_spiel.python.algorithms.psro_v2.quiesce.quiesce import PSROQuiesceSolver
from open_spiel.python.algorithms.psro_v2.quiesce import quiesce_sparse
from open_spiel.python.algorithms.nash_solver import general_nash_solver as gs
from open_spiel.python.algorithms.psro_v2.meta_strategies import general_nash_strategy

np.random.seed(4)
np.set_printoptions(precision=3)
max_player = 4 # maximum num of player in a game
min_player = 4 # minimum num of player in a game
test_cases = 5 # num test case to generate in total
min_policy = 3 # min number policy each agent has
max_policy = 5# max number of poicy each agent has
payoff_scale = 10  # payoff range [-payoff_scale,payoff_scale]
zero_threshold = 5e-3 # l1 difference betweent two vectors so that they are deemed different
result_folder = './test_all_eq_result'
result_file = result_folder +'/player_'+str(max_player)+'_'+str(min_player)+'_policies_'+str(max_policy)+'_'+str(min_policy)+'.txt'



def generate_parameters():
  player_num =  np.random.randint(min_player,max_player+1)
  params = {
    'player_num': player_num,
    'policy_num': np.random.randint(low=min_policy, high=max_policy+1, size=player_num).tolist(),
    'payoff_scale': payoff_scale
  }
  return params

def generate_meta_game(player_num, policy_num, payoff_scale):
  """
  Generate a payoff matrix in the form of full _meta_games
  Params:
    policy_num : a one dimensional list containing number of policy each player has
  """
  assert player_num==len(policy_num)
  meta_games = []
  for i in range(player_num):
    meta_games.append(payoff_scale*2*(np.random.rand(*policy_num)-0.5))
  return meta_games

def element_distance_to_set(ele,ele_set):
  """
  Calculate the minimum distance between an element and a set of element.
  Params:
    ele    : one nash equilibrium. A len(num_players) list of list. 
    ele_set: many ele. One dimensional more
  """
  min_dis = 1e5
  closest_index = 0
  for i in range(len(ele_set)):
    cur_dis = 0
    for j in range(len(ele)):
      cur_dis += np.linalg.norm(ele[j]-ele_set[i][j],ord=1)
    if cur_dis < min_dis:
      min_dis = cur_dis
      closest_index = i
  return closest_index, min_dis 

# Implement child classes for Quiesce, to override update agents, sampling payoff using policy. I am being very lenient in terms of how the codes are written.
class QuiesceTest(PSROQuiesceSolver):

  def __init__(self, meta_games):
    self._full_meta_game = meta_games
    self._game_num_players = len(meta_games)
    self._num_players = len(meta_games)
    self.symmetric_game = False
    self._sims_per_entry = 1
    self._initialize_policy()
    self._initialize_game_state()
    self._meta_strategy_method = general_nash_strategy

  def _initialize_policy(self):
    shape = self._full_meta_game[0].shape
    self._policies = [range(ele) for ele in shape]
    self._complete_ind = []
    for i in range(len(shape)):
      self._complete_ind.append([1]+[0 for _ in range(shape[i]-1)])

  def _initialize_game_state(self):
    # manually set the initial know component
    self._meta_games = []
    for ele in self._full_meta_game:
      temp = np.empty(ele.shape)
      temp[:] = np.nan
      first_ind = [0 for _ in range(len(ele.shape))]
      temp[tuple(first_ind)] = ele[tuple(first_ind)]
      self._meta_games.append(temp)
  
  def sample_episodes(self, policy, sims_per_entry=None):
    """
    Override the parent class of sampling episodes. Directly return the value in self._full_meta_game
    """
    return np.array([self._full_meta_game[i][tuple(policy)] for i in range(self._game_num_players)])

class QuiesceSparseTest(quiesce_sparse.PSROQuiesceSolver):
  def __init__(self, meta_games):
    self._full_meta_game = meta_games
    self._game_num_players = len(meta_games)
    self._num_players = len(meta_games)
    self.symmetric_game = False
    self._sims_per_entry = 1
    self._initialize_policy()
    self._initialize_game_state()
    self._meta_strategy_method = general_nash_strategy

  def _initialize_policy(self):
    shape = self._full_meta_game[0].shape
    self._policies = [range(ele) for ele in shape]
    self._complete_ind = []
    for i in range(len(shape)):
      self._complete_ind.append([1]+[0 for _ in range(shape[i]-1)])

  def _initialize_game_state(self):
    # manually set the initial know component
    self._meta_games = quiesce_sparse.sparray(self._game_num_players)
    policy = tuple([0 for _ in range(self._game_num_players)])
    data = [self._full_meta_game[i][policy] for i in range(self._game_num_players)]
    self._meta_games[policy] = data
  
  def sample_episodes(self, policy, sims_per_entry=None):
    """
    Override the parent class of sampling episodes. Directly return the value in self._full_meta_game
    """
    return np.array([self._full_meta_game[i][tuple(policy)] for i in range(self._game_num_players)])

def main():
  #sys.stdout = open(result_file,'w')
  #anomaly_meta_game_file = open(result_folder+'/anomaly_game_matrix.pkl','ab')
  for i in range(test_cases):
    params = generate_parameters()
    meta_game = generate_meta_game(**params)
    print("###################################")
    print("##############Test_"+str(i)+"_##############")
    print("###################################")
    print(params)

    print("-----------------Gambit_",end='')
    start = time.time()
    gambit_eq = gs.nash_solver(meta_game,solver="gambit",mode="all")
    end = time.time()
    print("{0:0.3f}".format(end-start),end='')
    print("--------------------")
    for eq in gambit_eq:
      print("%%%%%%%%%%%%%%%%%%% EQ %%%%%%%%%%%%%%%%%")
      for eq_q in eq:
        print(["{0:0.3f}".format(i) for i in eq_q.tolist()])
    
    # TODO: RD may be performing poorly because gambit-gnm fails to find all
    # equilibrium. Chang gambit to lca to test
    print("------------------RD_",end='')
    start = time.time()
    rd_eq = gs.nash_solver(meta_game,solver="replicator",mode='one')
    end = time.time()
    print("{0:0.3f}".format(end-start),end='')
    print("-----------------------")
    for ele in rd_eq:
      print(["{0:0.3f}".format(i) for i in ele])
    closest,min_dis = element_distance_to_set(rd_eq,gambit_eq)
    print('distance to gambit-eq',closest,"{0:0.3f}".format(min_dis))

    print("----------------QuieFul_",end='')
    quiesce_full = QuiesceTest(meta_game)
    start = time.time()
    quiesce_full_eq,_ = quiesce_full.inner_loop()
    end = time.time()
    print("{0:0.3f}".format(end-start),end='')
    print("-------------------")
    print(quiesce_full_eq)
    closest,min_dis = element_distance_to_set(quiesce_full_eq, gambit_eq)
    print('distance to gambit-eq',closest,"{0:0.3f}".format(min_dis))
    if min_dis > zero_threshold:
      #pickle.dump(meta_game, anomaly_meta_game_file)
      print(meta_game)

    print("----------------QuieSpa_",end='')
    quiesce_sparse = QuiesceSparseTest(meta_game)
    start = time.time()
    quiesce_sparse_eq,_ = quiesce_sparse.inner_loop()
    end = time.time()
    print("{0:0.3f}".format(end-start),end='')
    print("-------------------")
    print(quiesce_sparse_eq)
    closest,min_dis = element_distance_to_set(quiesce_sparse_eq, gambit_eq)
    print('distance to gambit-eq',closest,"{0:0.3f}".format(min_dis))

  #anomaly_meta_game_file.close()  

if __name__ == '__main__':
  main()

