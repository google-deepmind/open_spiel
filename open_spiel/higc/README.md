# Code related to the [Hidden Information Games Competition](http://higcompetition.info/).

There is an implementation of:

-   Random bots in [Python](./bots/random_bot.py) or
    [C++](./bots/random_bot.cc).
-   [Referee](./referee.h) that communicates with the bots (C++)
-   [Tournament](./tournament.cc) organized by the referee according to the
    rules of the competition (C++).

You can just copy-paste the random bots into your codebase and start developing
your own bot submission for the competition.

Follow instructions in the next section if you'd like test your bot locally in a
tournament setting.

## Set-up instructions with Python

Go to the `open_spiel/higc/bots` directory in your terminal and run interactive 
python console `ipython`. Then copy following snippet of code:

```
import pyspiel
referee = pyspiel.Referee("kuhn_poker", ["./random_bot.py", "./random_bot.py"])
results = referee.play_tournament(num_matches=1)
```

Then you should get an output similar to the following:

```
Starting players.
Bot#0: ./random_bot.py
Bot#1: ./random_bot.py
Bot#1: kuhn_poker 1
Bot#0: kuhn_poker 0
Bot#0 ready ok.
Bot#1 ready ok.

--------------------------------------------------------------------------------
Playing match 1 / 1
--------------------------------------------------------------------------------
Bot#0 start ok.
Bot#1 start ok.

History: 
Bot#1: AQM= AQI=
Bot#0: AQM= AQE=
Bot#0 ponder ok.
Bot#1 ponder ok.
Submitting actions: -1 -1
Chance action: 2 with prob 0.333333

History: 2
Bot#1: AQM= AQI=
Bot#0: AQM= ARE=
Bot#0 ponder ok.
Bot#1 ponder ok.
Submitting actions: -1 -1
Chance action: 1 with prob 0.5

History: 2 1
Bot#1: AQM= AQo=
Bot#0: AQM= ARE= 0 1
Bot#1 ponder ok.
Bot#0 act response: '1'
Bot#0 act ok. 
Submitting actions: 1 -1

History: 2 1 1
Bot#1: AAAAAEAAAIA/ AQo= 0 1
Bot#0: AAAAAEAAAIA/ ARE=
Bot#0 ponder ok.
Bot#1 act response: '0'
Bot#1 act ok. 
Submitting actions: -1 0

Match over!
History: 2 1 1 0
Bot#0 returns 1
Bot#0 protocol errors 0
Bot#0 illegal actions 0
Bot#0 ponder errors 0
Bot#0 time overs 0
Bot#1 returns -1
Bot#1 protocol errors 0
Bot#1 illegal actions 0
Bot#1 ponder errors 0
Bot#1 time overs 0
Bot#1: match over -1
score: -1
Bot#0: match over 1
score: 1
Bot#0 match over ok.
Bot#1 match over ok.

--------------------------------------------------------------------------------
Tournament is over!
--------------------------------------------------------------------------------
In total played 1 matches.
Average length of a match was 4 actions.

Corruption statistics:
Bot#0: 0
Bot#1: 0

Returns statistics:
Bot#0 mean: 1 var: 0
Bot#1 mean: -1 var: 0
Waiting for tournament shutdown (100ms)
Bot#1: tournament over
Bot#0: tournament over
Shutting down players.
```

For the same tournament settings as for HIGC, use following:

```
settings=pyspiel.TournamentSettings(timeout_ready = 5000,
                                    timeout_start = 200,
                                    timeout_act = 5000,
                                    timeout_ponder = 200,
                                    timeout_match_over = 1000,
                                    time_tournament_over = 60000,
                                    max_invalid_behaviors = 3,
                                    disqualification_rate = 0.1)
```

The code supports also more than two players (needs to give more time to setup
the players):
```
referee = pyspiel.Referee("goofspiel(players=10)", ["./random_bot.py"]*10,
                          settings=pyspiel.TournamentSettings(timeout_ready = 1000))
results = referee.play_tournament(num_matches=1)
```

## Set-up instructions with C++

First, follow [OpenSpiel install instructions](../../docs/install.md) for
installation from source and run all tests. As part of the test suite, there are
also tests for the competition (`referee_test.cc`) that should pass.

Then run the tournament in the console: 

```
$ # Set your own path 
$ OPEN_SPIEL_REPO=/home/michal/Code/open_spiel/ 
$ # Go to your build directory 
$ cd $OPEN_SPIEL_REPO/build/higc 
$ # Note the bots are located outside of the build directory!
$ ./tournament --game="kuhn_poker" \
--num_matches=1 \
--executables="$OPEN_SPIEL_REPO/open_spiel/higc/bots/random_bot_py.sh,$OPEN_SPIEL_REPO/open_spiel/higc/bots/random_bot_cpp.sh"
```

You should get an output similar to the output in the previous section.
