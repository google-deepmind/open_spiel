import numpy as np
from open_spiel.python.algorithms.nash_solver import subproc
import os
import pickle
import itertools
import logging

"""
This script connects meta-games with gambit. It translates a meta-game to a payoff matrix format 
that gambit could recognize.
Gambit file format: http://www.gambit-project.org/gambit16/16.0.0/formats.html 
"""

def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)

def save_pkl(obj,path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result

# Path that saves the payoff matrix for gamebit.
gambit_DIR = os.path.abspath(os.path.dirname(__file__)) + '/nfg/payoffmatrix.nfg'

# This functions help to translate meta_games into gambit nfg.
def product(shape, axes):
    prod_trans = tuple(zip(*itertools.product(*(range(shape[axis]) for axis in axes))))

    prod_trans_ordered = [None] * len(axes)
    for i, axis in enumerate(axes):
        prod_trans_ordered[axis] = prod_trans[i]
    return zip(*prod_trans_ordered)

def encode_gambit_file(meta_games):
    """
    Encode a meta-game to nfg file that gambit can recognize.
    :param meta_games: A meta-game (payoff tensor) in PSRO.
    """
    num_players = len(meta_games)
    # Write header
    with open(gambit_DIR, "w") as nfgFile:
        nfgFile.write('NFG 1 R "Empirical Game"\n')
        name_players = '{ "p1"'
        for i in range(2,num_players+1):
            name_players += " " + "\"" + 'p' + str(i) + "\""
        name_players += ' }'
        nfgFile.write(name_players)
        # Write strategies
        num_strs = '{ '
        range_iterators = []
        for i in np.shape(meta_games[0]):
            num_strs += str(i) + " "
            range_iterators += [range(i)]
        num_strs += '}'
        nfgFile.write(num_strs + '\n\n')
        # Write outcomes
        axes = tuple(reversed(range(num_players)))
        for current_index in product(np.shape(meta_games[0]), axes):
            for meta_game in meta_games:
                nfgFile.write(str(meta_game[tuple(current_index)]) + " ")

def gambit_analysis(timeout, method="gnm"):
    """
    Call a subprocess and run gambit to find all NE.
    :param timeout: Maximum time for the subprocess.
    :param method: The gamebit command line method.
    """
    if not isExist(gambit_DIR):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-" + method + " -q " + os.path.abspath(os.path.dirname(__file__)) + "/nfg/payoffmatrix.nfg -d 8 > " + os.getcwd() + "/nfg/nash.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)

def gambit_analysis_pure(timeout, method="enumpure"):
    """
    Call a subprocess and run gambit to find pure NE.
    :param timeout: Maximum time for the subprocess.
    """
    if not isExist(gambit_DIR):
        raise ValueError(".nfg file does not exist!")
    command_str = "gambit-" + method + " -q " + os.path.abspath(os.path.dirname(__file__)) + "/nfg/payoffmatrix.nfg > " + os.getcwd() + "/nfg/nash.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)

def decode_gambit_file(meta_games, mode="all", max_num_nash=10):
    """
    Decode the results returned from gambit to a numpy format used for PSRO.
    :param meta_games: A meta-game in PSRO.
    :param mode: "all", "pure", "one" options
    :param max_num_nash: the number of NE considered to return
    :return: a list of NE
    """
    nash_DIR = os.getcwd() + '/nfg/nash.txt'
    if not isExist(nash_DIR):
        raise ValueError("nash.txt file does not exist!")
    num_lines = file_len(nash_DIR)

    logging.info("Number of NE is ", num_lines)
    if max_num_nash != None:
        if num_lines >= max_num_nash:
            num_lines = max_num_nash
            logging.info("Number of NE is constrained by the num_nash.")

    shape = np.shape(meta_games[0])
    slice_idx = []
    pos = 0
    for i in range(len(shape)):
        slice_idx.append(range(pos, pos + shape[i]))
        pos += shape[i]

    equilibria = []
    with open(nash_DIR,'r') as f:
        for _ in np.arange(num_lines):
            equilibrim = []
            nash = f.readline()
            if len(nash.strip()) == 0:
                continue
            nash = nash[3:]
            nash = nash.split(',')
            new_nash = []
            for j in range(len(nash)):
                new_nash.append(convert(nash[j]))

            new_nash = np.array(new_nash)
            new_nash = np.round(new_nash, decimals=8)
            for idx in slice_idx:
                equilibrim.append(new_nash[idx])
            equilibria.append(equilibrim)

    if mode == "all" or mode == "pure":
        return equilibria
    elif mode == "one":
        return equilibria[0]
    else:
        logging.info("mode is beyond all/pure/one.")


def do_gambit_analysis(meta_games, mode, timeout = 600, method="gnm", method_pure_ne="enumpure"):
    """
    Combine encoder and decoder.
    :param meta_games: meta-games in PSRO.
    :param mode: "all", "pure", "one" options
    :param timeout: Maximum time for the subprocess
    :param method: The gamebit command line method.
    :param method_pure_ne: The gamebit command line method for finding pure NE.
    :return: a list of NE.
    """
    encode_gambit_file(meta_games)
    while True:
        if mode == 'pure':
            gambit_analysis_pure(timeout, method_pure_ne)
        else:
            gambit_analysis(timeout, method)
        # If there is no pure NE, find mixed NE.
        nash_DIR = os.path.abspath(os.path.dirname(__file__)) + '/nfg/nash.txt'
        if not isExist(nash_DIR):
            raise ValueError("nash.txt file does not exist!")
        num_lines = file_len(nash_DIR)
        if num_lines == 0:
            logging.info("Pure NE does not exist. Return mixed NE.")
            mode = 'all'
            continue
        equilibria = decode_gambit_file(meta_games, mode)
        if len(equilibria) != 0:
            break
        timeout += 120
        if timeout > 7200:
            logging.info("Gambit has been running for more than 2 hour.!")
        logging.info("Timeout has been added by 120s.")
    logging.info('gambit_analysis done!')
    return equilibria


def convert(s):
    """
    Convert a probability string to a float number.
    :param s: probability string.
    :return: a float probability.
    """
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def file_len(fname):
    """
    Get the number of lines in a text file. This is used to count the number of NE found by gamebit.
    :param fname: file name.
    :return: number of lines.
    """
    num_lines = sum(1 for line in open(fname))
    return num_lines



