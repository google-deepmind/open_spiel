import pyspiel
game = pyspiel.load_game('bridge(use_double_dummy_result=false)')

line = '30 32 10 35 50 45 21 7 1 42 39 43 0 16 40 20 36 15 22 44 26 6 4 51 47 46 25 14 29 5 34 11 49 31 37 9 41 13 24 8 28 17 48 23 33 18 3 19 38 2 27 12 56 57 52 63 52 52 52 0 32 48 8 3 51 47 15 44 28 16 4 14 50 2 10 49 5 37 9 36 31 24 20 46 22 12 26 13 25 19 1 43 41 17 27 7 33 45 39 40 23 29 6 11 30 18 21 35 38 42 34'
actions = (int(x) for x in line.split(' '))
state = game.new_initial_state()
for a in actions: state.apply_action(a)
print(state)
