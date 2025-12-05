import numpy as np
from numpy.polynomial import Polynomial as P
from kosaraju import *
import sympy as sp
from sympy import *
from sympy.abc import t
from sympy.functions.elementary.piecewise import Undefined
from collections import deque
import pyspiel

# ===============================
# SSD ALGORITHM from John
# ===============================

def stableDistribution(M):
    """Computes the normalized eigenvector for eigenvalue 1 for a unichain Markov matrix."""
    (eigs, vects) = np.linalg.eig(M)
    idx = (eigs == eigs.max()).nonzero()
    v = np.real(np.transpose(vects)[idx][0])
    return normalize(v)

def polyPrint(arg):
    """A helper function for printing polynomials."""
    return sp.Poly(arg.coef, t).as_expr()

print_polyArray = np.frompyfunc(polyPrint, 1, 1)

def polyEval(arg, val):
    """A helper function for evaluating polynomials."""
    return np.polyval(arg, val) if isinstance(arg, np.poly1d) else arg

evalPolyArray = np.frompyfunc(polyEval, 2, 1)

def zeroTest(x):
    """A helper function for testing a scalar against 0."""
    return x == 0

def polyZeroTest(p):
    """A helper function for testing a polynomial against 0."""
    return p == np.poly1d([0])

def shiftExponent(p, degree):
    """A helper function for lowering the degree of a polynomial."""
    coefs = list(p.c)
    p_degree = len(coefs) - 1
    if (p_degree < degree):
        return np.poly1d([])
    del coefs[p_degree - degree + 1 : p_degree + 1]
    return np.poly1d(coefs)

def costResistance(p):
    """A helper function for computing the cost and resistance."""
    coefs = p.c
    l = len(coefs)
    if (l == 1 and (0 == coefs[0])):
        return {'cost': 1, 'resistance': 'infinity'}
    i = l - 1
    while (i > 0) and (coefs[i] == 0):
        i -= 1
    return {'cost': coefs[i], 'resistance': l - i - 1}

def hasZeroOnDiagonalP(mat, zeroTest):
    """A helper function for testing whether a matrix has a zero on its diagonal."""
    result = False
    for col in range(mat.shape[1]):
        if (zeroTest(mat[col, col])):
            result = True
    return result

def uniformScale(mat, zeroTest):
    """A helper function for applying a default uniform scaling factor to a PMM, if necessary."""
    if (not hasZeroOnDiagonalP(mat, zeroTest)):
        return mat
    poly_one = np.poly1d([1])
    return 0.5 * (mat + poly_one*np.identity(mat.shape[0]))

def dropNonZeroR(alg):
    """A helper function for adjusting the supplied matrix."""
    return alg

def nonUniformScale(mat):
    """If possible, applies a non-uniform scaling to mat."""
    result = {}
    result['dim'] = mat.shape[1] 
    dim = range(result['dim'])
    
    minResistance = -1
    maxCostSum = 0
    nonTransientCols = list(dim)
    
    for col in dim:
        minColResistance = -1
        colCostSum = 0
        for row in dim:
            if (row == col):
                continue
            cr = costResistance(mat[row, col])
            if (cr['resistance'] == 'infinity'):
                continue
            if (cr['resistance'] == 0):
                minColResistance = 0
                break
            elif (minColResistance < 0 or cr['resistance'] < minColResistance):
                minColResistance = cr['resistance']
                colCostSum = cr['cost']
            elif (cr['resistance'] == minColResistance):
                colCostSum += cr['cost']

        if (minColResistance == 0):
            nonTransientCols.remove(col)
        elif (minColResistance > 0):
            if ((minResistance < 0) or (minColResistance < minResistance)):
                minResistance = minColResistance
                maxCostSum = colCostSum
            elif (minColResistance == minResistance):
                maxCostSum += colCostSum

    if (len(nonTransientCols) == 0):
        raise Exception("No scaling is possible.")
 
    result['mat'] = np.array(mat, copy=True) 
    new_mat = result['mat']
    result['D'] = np.array(np.identity(result['dim']), dtype=object) 
    Dmatrix = result['D']
    del result['dim']

    f_cost = 2 * maxCostSum
    tmp = [f_cost]
    tmp.extend([0] * minResistance)
    f =  np.poly1d(tmp)
    for col in dim:
        if (col not in nonTransientCols):
            Dmatrix[col, col] = f        

    poly_one = np.poly1d([1])
    for col in dim:
        if (col in nonTransientCols):
            for row in dim:      
                if (row == col):
                    new_mat[row, col] = poly_one + (poly_one/f_cost) * shiftExponent(mat[row, col] - poly_one, minResistance)
                else:
                    new_mat[row, col] = (poly_one/f_cost) * shiftExponent(mat[row, col], minResistance)
    return result

def reduce(mat, s, M0 = None):
    """Eliminates all the states of mat in s."""
    result = {}
    if (len(s) == 0):
        raise Exception("s cannot be empty.")
    
    dim = mat.shape[1]
    if M0 is None:
        M0 = np.array(evalPolyArray(mat,0), dtype=float)

    sbar = list(set(range(dim)).symmetric_difference(set(s)))
    perm = list(sbar)
    perm.extend(s)
    P = 0*np.identity(dim)
    for col in range(dim):
        P[perm[col], col] = 1

    Mssbar = mat[np.ix_(s, sbar)]
    LambdassInv = np.linalg.inv(M0[np.ix_(s, s)] - np.identity(len(s)))
    result['i'] = P@np.block([[np.identity(len(sbar))], [-LambdassInv@Mssbar]]) 
    tmp = mat[np.ix_(sbar, sbar)] - mat[np.ix_(sbar, s)]@LambdassInv@Mssbar

    dim = tmp.shape[1]
    zero = np.poly1d([0])
    one = np.poly1d([1])
    for col in range(dim):
        sum = zero
        for row in range(dim):
            if (row == col):
                continue
            cr = costResistance(tmp[row, col])
            if cr['resistance'] == 'infinity':
                tmp[row, col] = zero
            else: 
                tmp[row, col] = [0] * (cr['resistance'] + 1)
                tmp[row, col][0] = cr['cost']
                tmp[row, col] = np.poly1d(tmp[row, col])
            sum = sum + tmp[row, col]
        tmp[col, col] = one - sum

    result['mat'] = tmp
    return result

def normalize(v):
    """Divides a vector by its sum, so it sums to 1.""" 
    return v/np.sum(v)

def SSD_step(mat):
    """Performs the next possible reduction step in the SSD algorithm."""
    result = {}
    M0 = np.array(evalPolyArray(mat,0), dtype=float)
    G = Graph(M0, zeroTest)
    communicating = G.CommunicatingClasses()
 
    if (len(communicating) == 1):
        result['stab'] = stableDistribution(M0)
        return result

    closed = G.ClosedClasses()
    maxSize = 0
    maxClass = None
    numNonTrivial = 0
    for x in closed:
        classLen = len(closed[x])
        if classLen > 1:
            numNonTrivial = numNonTrivial + 1
        if classLen > maxSize:
            maxSize = classLen
            maxClass = closed[x]
 
    s = list(maxClass)
    s.pop(0)
    if (numNonTrivial > 0):
        return reduce(mat, s, M0)
 
    return nonUniformScale(mat)

def SSD_iterate(mat):
    """Recursively apply SSD_step to compute the stable distribution."""
    result = SSD_step(mat)

    if ('stab' in result):
        return result['stab']

    if ('i' in result):
        return result['i']@SSD_iterate(result['mat'])
    
    return result['D']@SSD_iterate(result['mat'])

def SSD(mat):
    """Computes the SSD of a PMM."""
    g = Graph(mat, polyZeroTest)
    if len(g.CommunicatingClasses().keys()) > 1:
        raise Exception("Input must be unichain.")

    stab = SSD_iterate(mat)
    return normalize(np.array(evalPolyArray(stab,0), dtype=float))

# ========================================
# ADAPTIVE LEARNING SIMULATION  
# ========================================

def get_payoff_matrices(game):
    """Extract payoff matrices from PySpiel game."""
    n_actions = game.num_distinct_actions()
    num_players = game.num_players()
    
    game1 = np.zeros((n_actions, n_actions))
    game2 = np.zeros((n_actions, n_actions))
    
    state = game.new_initial_state()
    
    if state.is_simultaneous_node():
        for action1 in range(n_actions):
            for action2 in range(n_actions):
                temp_state = game.new_initial_state()
                temp_state.apply_actions([action1, action2])
                
                if temp_state.is_terminal():
                    returns = temp_state.returns()
                    game1[action1, action2] = returns[0]
                    if num_players > 1:
                        game2[action2, action1] = returns[1]
    
    return game1, game2

def joint_state_to_index(history1, history2, n_actions, m):
    """Convert joint histories to state index."""
    if len(history1) < m or len(history2) < m:
        return 0
    
    index1 = 0
    for i, action in enumerate(history1[-m:]):
        index1 += action * (n_actions ** (m - 1 - i))
    
    index2 = 0
    for i, action in enumerate(history2[-m:]):
        index2 += action * (n_actions ** (m - 1 - i))
    
    return index1 + index2 * (n_actions ** m)

def index_to_joint_state(index, n_actions, m):
    """Convert state index back to joint histories."""
    n_states_per_player = n_actions ** m
    
    index2 = index // n_states_per_player
    index1 = index % n_states_per_player
    
    history1 = []
    temp_index1 = index1
    for i in range(m):
        history1.append(temp_index1 % n_actions)
        temp_index1 //= n_actions
    history1 = list(reversed(history1))
    
    history2 = []
    temp_index2 = index2
    for i in range(m):
        history2.append(temp_index2 % n_actions)
        temp_index2 //= n_actions
    history2 = list(reversed(history2))
    
    return history1, history2

def compute_best_response_probs(game_matrix, opponent_mixed_strategy):
    """Compute probability distribution over best responses."""
    expected_payoffs = game_matrix @ opponent_mixed_strategy
    max_payoff = np.max(expected_payoffs)
    best_actions = np.where(expected_payoffs == max_payoff)[0]
    
    action_probs = np.zeros(len(expected_payoffs))
    action_probs[best_actions] = 1.0 / len(best_actions)
    
    return action_probs

def compute_best_response(game_matrix, opponent_mixed_strategy, epsilon=0.0):
    """Compute best response to opponent's mixed strategy."""
    expected_payoffs = game_matrix @ opponent_mixed_strategy
    max_payoff = np.max(expected_payoffs)
    best_actions = np.where(expected_payoffs == max_payoff)[0]
    
    action_probs = np.zeros(len(expected_payoffs))
    
    if epsilon == 0:
        action_probs[best_actions] = 1.0 / len(best_actions)
    else:
        action_probs += epsilon / len(action_probs)
        action_probs[best_actions] += (1 - epsilon) / len(best_actions)
    
    action = np.random.choice(len(action_probs), p=action_probs)
    return action, action_probs

def construct_perturbed_markov_matrix(game, m, s):
    """Construct the perturbed Markov matrix for adaptive learning dynamics."""
    game1, game2 = get_payoff_matrices(game)
    n_actions = game.num_distinct_actions()
    n_states = (n_actions ** m) ** 2
    
    # Create symbolic variable for epsilon
    x = np.poly1d([1, 0])  # epsilon
    p_one = np.poly1d([1])  # 1
    p_zero = np.poly1d([0])  # 0
    
    M = np.zeros((n_states, n_states), dtype=object)
    
    for state_idx in range(n_states):
        hist_p0_sees_p1, hist_p1_sees_p0 = index_to_joint_state(state_idx, n_actions, m)
        
        for next_state_idx in range(n_states):
            hist_p0_sees_p1_next, hist_p1_sees_p0_next = index_to_joint_state(next_state_idx, n_actions, m)
            
            transition_prob = p_zero
            
            for action_p0 in range(n_actions):
                for action_p1 in range(n_actions):
                    expected_hist_p0_sees_p1 = hist_p0_sees_p1[1:] + [action_p1] if m > 1 else [action_p1]
                    expected_hist_p1_sees_p0 = hist_p1_sees_p0[1:] + [action_p0] if m > 1 else [action_p0]
                    
                    if (expected_hist_p0_sees_p1 == hist_p0_sees_p1_next and 
                        expected_hist_p1_sees_p0 == hist_p1_sees_p0_next):
                        
                        if m == 0:
                            action_prob = (p_one / n_actions) * (p_one / n_actions)
                        else:
                            # Player 0's empirical distribution of Player 1's actions
                            pi_p1 = np.zeros(n_actions)
                            for a in hist_p0_sees_p1:
                                pi_p1[a] += 1.0 / len(hist_p0_sees_p1)
                            
                            # Player 1's empirical distribution of Player 0's actions  
                            pi_p0 = np.zeros(n_actions)
                            for a in hist_p1_sees_p0:
                                pi_p0[a] += 1.0 / len(hist_p1_sees_p0)
                            
                            # Best response probabilities
                            p0_best_responses = compute_best_response_probs(game1, pi_p1)
                            p1_best_responses = compute_best_response_probs(game2, pi_p0)
                            
                            # Action probabilities with experimentation
                            p0_action_prob = (p_one - x) * p0_best_responses[action_p0] + x / n_actions
                            p1_action_prob = (p_one - x) * p1_best_responses[action_p1] + x / n_actions
                            
                            action_prob = p0_action_prob * p1_action_prob
                        
                        transition_prob = transition_prob + action_prob
            
            M[next_state_idx, state_idx] = transition_prob
    
    return M

def adaptive_learning_stationary(game, m, s, T=100000, epsilon=0.1):
    """Simulate adaptive learning and return the stationary distribution."""
    game1, game2 = get_payoff_matrices(game)
    n_actions = game.num_distinct_actions()
    n_joint_states = (n_actions ** m) ** 2
    
    state_visits = np.zeros(n_joint_states)
    
    history_p0_sees_p1 = deque(maxlen=m)
    history_p1_sees_p0 = deque(maxlen=m)
    
    for i in range(T + m):
        if len(history_p0_sees_p1) == 0 or len(history_p1_sees_p0) == 0:
            action_p0 = np.random.randint(0, n_actions)
            action_p1 = np.random.randint(0, n_actions)
        else:
            sample_size_p0 = min(s, len(history_p0_sees_p1))
            sample_size_p1 = min(s, len(history_p1_sees_p0))
            
            p0_sampled_actions = np.random.choice(list(history_p0_sees_p1), sample_size_p0, replace=False)
            p1_sampled_actions = np.random.choice(list(history_p1_sees_p0), sample_size_p1, replace=False)
            
            pi_p1 = np.histogram(p0_sampled_actions, bins=range(n_actions + 1))[0] / sample_size_p0
            pi_p0 = np.histogram(p1_sampled_actions, bins=range(n_actions + 1))[0] / sample_size_p1
            
            action_p0, _ = compute_best_response(game1, pi_p1, epsilon)
            action_p1, _ = compute_best_response(game2, pi_p0, epsilon)
        
        history_p0_sees_p1.append(action_p1)
        history_p1_sees_p0.append(action_p0)
        
        if i >= m and len(history_p0_sees_p1) == m and len(history_p1_sees_p0) == m:
            joint_state_idx = joint_state_to_index(list(history_p0_sees_p1), list(history_p1_sees_p0), n_actions, m)
            state_visits[joint_state_idx] += 1
    
    return state_visits / np.sum(state_visits)


def compare_ssd_and_simulation(game, m, s, T=10000, epsilon=0.01):
    """Compare SSD analytical result with simulation.
    
    What fraction of the time does SSD say we'll be playing a strategy
    versus what fraction do we actually play that strategy    
    """
    print(f"Comparing SSD vs Simulation for m={m}, s={s}, epsilon={epsilon}")
    
    # Get SSD result (analytical)
    print("Computing SSD (analytical)...")
    try:
        M = construct_perturbed_markov_matrix(game, m, s)
        ssd_result = SSD(M)
    except Exception as e:
        print(f"SSD failed: {e}")
        return None
    
    # Get simulation result (empirical)
    print("Running simulation...")
    sim_result = adaptive_learning_stationary(game, m, s, T, epsilon)
    
    # Compare results
    n_actions = game.num_distinct_actions()
    n_joint_states = (n_actions ** m) ** 2
    
    print(f"\nResults comparison:")
    print(f"Number of joint states: {n_joint_states}")
    print(f"SSD result shape: {ssd_result.shape}")
    print(f"Simulation result shape: {sim_result.shape}")
    
    # Print state-by-state comparison
    print(f"\nState-by-state comparison:")
    print(f"{'State':<6} {'P0_hist':<10} {'P1_hist':<10} {'SSD':<10} {'Simulation':<10} {'Diff':<10}")
    print("-" * 70)
    
    max_diff = 0
    for state_idx in range(n_joint_states):
        hist1, hist2 = index_to_joint_state(state_idx, n_actions, m)
        ssd_prob = ssd_result[state_idx]
        sim_prob = sim_result[state_idx]
        diff = abs(ssd_prob - sim_prob)
        max_diff = max(max_diff, diff)
        
        if ssd_prob > 0.01 or sim_prob > 0.01:  # Only show significant states
            print(f"{state_idx:<6} {str(hist1):<10} {str(hist2):<10} {ssd_prob:<10.4f} {sim_prob:<10.4f} {diff:<10.4f}")
    
    # Summary statistics
    mse = np.mean((ssd_result - sim_result) ** 2)
    mae = np.mean(np.abs(ssd_result - sim_result))
    
    print(f"\nSummary:")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute error (MAE): {mae:.6f}")
    print(f"Mean squared error (MSE): {mse:.6f}")
    print(f"Correlation: {np.corrcoef(ssd_result, sim_result)[0,1]:.6f}")
    
    return {
        'ssd_result': ssd_result,
        'sim_result': sim_result,
        'max_diff': max_diff,
        'mae': mae,
        'mse': mse,
        'correlation': np.corrcoef(ssd_result, sim_result)[0,1]
    }

def main():
    """Main function to run the comparison."""
    # Load matching pennies game
    # game = pyspiel.load_game('matrix_mp')
    game = pyspiel.load_game('matrix_bos')
    
    # Test with different parameters
    test_cases = [
        (1, 1),  # Small case
        (2, 1),  # Medium case
        (2, 2),  # Larger case
        # (4, 2)
    ]
    
    for m, s in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing m={m}, s={s}")
        print('='*60)
        
        results = compare_ssd_and_simulation(game, m, s, T=100000, epsilon=0.001)
            

if __name__ == "__main__":
    main()