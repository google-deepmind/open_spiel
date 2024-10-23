'''This is where you will test your algorithm against other players'''

import random
import numpy as np

from open_spiel.python.games import gt3

def playGT3():
    game = gt3.GT3Game()
    state = game.new_initial_state()
    print(state) # display board
    while not state.is_terminal():
        print("Player {}'s turn".format(state.current_player()))
        print("Legal moves:", state.legal_actions())
        action = np.random.choice(state.legal_actions())
        state.apply_action(action) # make move
        print(state, '\n')

    print("Returns: {}".format(state.returns()))
    if state.returns()[0] == 1:
       print("Player 0 wins!")
       return 0
    elif state.returns()[1] == 1:
       print("Player 1 wins!")
       return 1
    else:
       print("This game is a tie!")
       return -1

if __name__ == "__main__":
    while True:
        try:
            gens = int(input('Enter number of generations to play: '))
            if gens < 1:
                raise ValueError
            break
        except ValueError:
            print('Please enter a positive integer')
    
    player_0_wins = 0
    player_1_wins = 0
    draws = 0

    for i in range(gens):
        winner = playGT3()
        if winner == 0:
            player_0_wins += 1
        elif winner == 1:
            player_1_wins += 1
        else:
            draws += 1
    
    print("\nGame Over!\n")
    print("Player 0 wins:", player_0_wins)
    print("Player 1 wins:", player_1_wins)
    print("Draws", draws)
