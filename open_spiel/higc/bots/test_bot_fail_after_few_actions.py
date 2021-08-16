# A bot that picks the first action from the list for the first two rounds,
# and then exists with an exception.
# Used only for tests.

game_name = input()
play_as = int(input())
print("ready")

while True:
    print("start")
    num_actions = 0
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
        if should_act:
            num_actions += 1
            print(legal_actions[-1])
        else:
            print("ponder")

        if num_actions > 2:
            raise RuntimeError
