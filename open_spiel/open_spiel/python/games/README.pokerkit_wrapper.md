# PokerkitWrapper and RepeatedPokerkit

As per its name, PokerkitWrapper wraps [PokerKit](https://github.com/uoftcprg/pokerkit)
to support multiple ["pre-defined" poker game variants](https://pokerkit.readthedocs.io/en/stable/simulation.html#pre-defined-games).

For more details see the following paper:

```
@ARTICLE{10287546,
  author={Kim, Juho},
  journal={IEEE Transactions on Games},
  title={PokerKit: A Comprehensive Python Library for Fine-Grained Multivariant Poker Game Simulations},
  year={2025},
  volume={17},
  number={1},
  pages={32-39},
  keywords={Games;Libraries;Automation;Artificial intelligence;Python;Computational modeling;Engines;Board games;card games;game design;games of chance;multiagent systems;Poker;rule-based systems;scripting;strategy games},
  doi={10.1109/TG.2023.3325637}}
```

RepeatedPokerkit itself wraps PokerkitWrapper (double-wrapping pokerkit) in
to support playing multiple hands within the same game episode, enabling
simulation of both cash games and tournaments.

We maintain separate playthrough files for each Pokerkit variant inside
https://github.com/deepmind/open_spiel/blob/master/open_spiel/integration_tests/playthroughs/.

