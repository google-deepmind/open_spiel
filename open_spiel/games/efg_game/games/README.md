This directory contains files that are in the
[Gambit](http://www.gambit-project.org/)
[extensive-form game (.efg) format](http://www.gambit-project.org/gambit14/formats.html).

To load them, use game string `efg_game(filename=<path to file>)`.
The parser is found in [efg_game.h](https://github.com/deepmind/open_spiel/blob/master/open_spiel/games/efg_game.h).

To export existing games in the library into gambit format, you can use python
script `python/examples/gambit_example.py`
