# Mean field games

This directory contains mean field games implemented in C++. For now, only
discrete-action, finite-state, single-population mean field games are
supported.

For reference on mean field games as well as common environment, please refer to:

* [Fictitious play for mean field games: Continuous time analysis and
applications", Perrin & al. 2019](https://arxiv.org/abs/2007.03458).

* [Scaling up Mean Field Games with Online Mirror Descent, Perolat & al. 2021](https://arxiv.org/pdf/2103.00623).


Typically, external logic will maintain:

* A probability distribution over game states representing an infinite number of
  identical and anonymous players.

* A finite collection of game state instances on the support of that
  distribution, implementing game dynamics and rewards.

Each game instance behaves similarly to a general-sum, perfect-information,
explicit-stochastic 1-player game, with the important difference that rewards
can depend on the whole state distribution.

Game states go through the following stages:

* The first game state is a chance node allowing sampling from the initial game
  state distribution.

Then game states cycle over:

1. Decision node with normal in-game actions (e.g. {left, neutral, right}).

2. Mean field node, where we expect that external logic will have update the
   state distribution and call DistributionSupport() and UpdateDistribution().

3. Chance node, where one of the normal in-game action (e.g. {left, neutral,
   right}) can be randomly selected.

