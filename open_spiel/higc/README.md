# Code related to the [Hidden Information Games Competition](http://higcompetition.info/).

There is an implementation of:

- Random bots in [Python](./bots/random_bot.py) or [C++](./bots/random_bot.cc).
- [Referee](./referee.h) that communicates with the bots (C++)
- [Tournament](./tournament.cc) organized by the referee according to the rules of the competition (C++).

You can just copy-paste the random bots into your codebase and start developing
your own bot submission for the competition.

Follow instructions in the next section if you'd like to setup the referee to
test your bot in a tournament setting.

## Set-up instructions

First, follow [OpenSpiel install instructions](../../docs/install.md) for
installation from source and run all tests. As part of the test suite, there are
also tests for the competition (`referee_test.cc`) that should pass.

Then run the tournament in the console:
```
$ # Set your own path
$ OPEN_SPIEL_REPO=/home/michal/Code/open_spiel/
$ # Go to your build directory
$ cd $OPEN_SPIEL_REPO/build/higc
$ ./tournament --game="kuhn_poker" --num_matches=1 --executables="$OPEN_SPIEL_REPO/open_spiel/higc/bots/random_bot_py.sh,$OPEN_SPIEL_REPO/open_spiel/higc/bots/random_bot_cpp.sh"
```

You should get an output similar to the following:

```
Starting players.
Bot#0: /home/michal/Code/open_spiel/open_spiel/higc/bots/random_bot_py.sh
Bot#1: /home/michal/Code/open_spiel/open_spiel/higc/bots/random_bot_cpp.sh
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
Bot actions: -1 -1
Chance action: 2 with prob 0.333333

History: 2
Bot#0: AQM= ARE=
Bot#1: AQM= AQI=
Bot#0 ponder ok.
Bot#1 ponder ok.
Bot actions: -1 -1
Chance action: 1 with prob 0.5

History: 2 1
Bot#0: AQM= ARE= 0 1
Bot#1: AQM= AQo=
Bot#1 ponder ok.
Bot#0 act response: '1'
Bot#0 act ok.
Bot actions: 1 -1

History: 2 1 1
Bot#0: AAAAAEAAAIA/ ARE=
Bot#1: AAAAAEAAAIA/ AQo= 0 1
Bot#0 ponder ok.
Bot#1 act response: '1'
Bot#1 act ok.
Bot actions: -1 1

Match over!
History: 2 1 1 1
Bot#0 returns 2
Bot#0 protocol errors 0
Bot#0 illegal actions 0
Bot#0 ponder errors 0
Bot#0 time overs 0
Bot#1 returns -2
Bot#1 protocol errors 0
Bot#1 illegal actions 0
Bot#1 ponder errors 0
Bot#1 time overs 0
Bot#0: match over 2
score: 2
Bot#1: match over -2
score: -2
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
Bot#0 mean: 2 var: 0
Bot#1 mean: -2 var: 0
Bot#1: tournament over
Bot#0: tournament over
Shutting down players.
```