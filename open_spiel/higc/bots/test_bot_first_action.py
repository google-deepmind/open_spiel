# A bot that picks the first action from the list. Used only for tests.

import base64
import pyspiel
import sys
from open_spiel.python.observation import make_observation

game_name = input()
play_as = int(input())
game = pyspiel.load_game(game_name)
public_observation = make_observation(
    game, pyspiel.IIGObservationType(perfect_recall=False, public_info=True,
                                     private_info=pyspiel.PrivateInfoType.NONE))
private_observation = make_observation(
    game, pyspiel.IIGObservationType(perfect_recall=False, public_info=False,
                                     private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER))
print("ready")

while True:
    print("start")
    while True:
        message = input()
        if message == "tournament over":
            print("tournament over")
            sys.exit(0)
        if message.startswith("match over"):
            print("match over")
            break
        public_buf, private_buf, *legal_actions = message.split(" ")
        should_act = len(legal_actions) > 0
        public_observation.decompress(base64.b64decode(public_buf))
        private_observation.decompress(base64.b64decode(private_buf))
        if should_act:
            print(legal_actions[0])
        else:
            print("ponder")
