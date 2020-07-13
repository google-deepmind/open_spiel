# Public State API (PS-API)

This is work-in-progress: many of the concepts / docs / descriptions are subject
to change. Some things might be even plain wrong, before we figure out all the
details!

This project is being developed as part of the
[OpenSpiel Extravaganza](https://github.com/deepmind/open_spiel/issues/251).

This document is best viewed with
[MathJax Plugin for Github](https://github.com/orsharir/github-mathjax/).

--------------------------------------------------------------------------------

<p align="center">
  <img src="docs/_static/public_tree_kuhn.png"
       alt="Visualization of Base API and Public State API for Kuhn Poker">
</p>

This directory contains Public State API: a special OpenSpiel sub-API that
enables implementation of algorithms that are used to efficiently solve large
games of imperfect information, primarily by taking advantage of value function
approximation by using neural networks. An example algorithm that can take
advantage of this API is [DeepStack](https://www.deepstack.ai/), or in general
any algorithm that uses continual resolving to compute strategies.

Within this directory we refer to the standard base classes of `State` / `Game`
in `open_spiel/spiel.h` as "Base API" and to base classes in this directory as
Public State API (PS-API).

We introduce PS-API because some algorithms require a special treatment of the
game structures in imperfect-information games. While these structures can be
deduced from Base API (see section on Game transformations), it is prohibitively
expensive to do so in general. It requires the construction of the whole game
tree, which is impossible for big games like Chess or HUNL Hold'em Poker.

Much of this API is based on the work of Factored-Observations Games (FOG) [1],
with special addition of *imperfect recall* abstractions. The reader is advised
to become familiar with [1], as it is out of scope of the README to address all
the subtle issues that arise in imperfect info games.

To understand this API, it is beneficial to understand how Base API relates to
FOGs. Base API does not implement FOGs exactly, it is rather a hybrid between
FOGs and EFGs. There are a number of subtle differences, significant for PS-API,
described in the file `fog_constants.h`.

The factored observations in `State`, accessed by `PrivateObservationString` or
`PublicObservationString` can be used to create the public state API via a game
transformation. What we call a "public state" is a subset (partitioning) of the
game tree, identified by public-observation history (i.e. `State`s which have
the same public-observation history).

In PS-API we use imperfect recall abstractions: many games involve some notion
of movement on a graph, where you can encounter the same positions repeatedly:
for example, in Chess the rooks can move around in circles. This results in
large number of identical recursive sub-games, which make solving these games
unnecessarily expensive (And without capping number of moves in the game,
infinite).

The imperfect recall abstraction allows us to group these situations together.
However, doing so in imperfect information games is more complicated. For the
purposes of the algorithms that find Nash equilibrium strategies in these games,
we can group together situations which have identical _value functions_.

While we may encounter the same board situations repeatedly, in a strict sense
they are not indeed the same: we may have received some observations that will
change our beliefs. However, these sub-games may have the same value function
(and therefore optimal policy, the goal we are after). Then all we need is to
encode the beliefs updates appropriately.

When computing strategies, there are two "phases" of a typical algorithm:
Top-Down and Bottom-Up traversal. During the Top-Down, we compute cumulative
products of the strategy probabilities, known also as reach probabilities or
`ReachProbs`. During Bottom-Up, we propagate the leaf values up the tree,
modifying the strategy we've come up with previously. For CFR-based algorithms,
we propagate the counter-factual values -- `CfValues`.

Counter-factual value of player $i$ at history $h$ with strategy profile
$\sigma$ is an expected value of the history $h$ multiplied by reach
probabilities of all other players (and chance) denoted as $j$, excluding $i$:

$ v_i^\sigma(h) = u_i^\sigma(h) \prod_{j} \pi_j^\sigma_j(h). $

Counter-factual action-value when taking action $a$ is defined similarly as
taking only the expected utility at child history $h.a$:

$ v_i^\sigma(h, a) = u_i^\sigma(h.a) \prod_{j} \pi_j^\sigma_j(h). $

**Counter-factual value of infoset** $I$ is

$ v_i^\sigma(I) = \sum_{h \in I} v_i^\sigma(h), $

and analogously the **counter-factual action-values** of infoset $I$ are:

$ v_i^\sigma(I, a) = \sum_{h \in I} v_i^\sigma(h, a). $

Especially note that $v_i^\sigma(h)$ includes a chance reach probability term.
The infoset values are sums over histories. However, the histories which are in
the same information set may have different chance reach probabilities,
complicating the design of PS-API.

There are two ways of how we can compute the CFVs of infosets:

1.  Maintain a collection of history cf. values, and sum them up as in the
    definition.
2.  Operate directly over cf. values of infosets, computing parent infoset
    values from the child values.

The second option is much more efficient in terms of memory and computation
needed, however it complicates how we need to treat chance probabilities.
Because each history within an infoset can have a different chance reach, we'd
have to keep them separately thus needing to maintain it over histories, which
we do not want. One way to deal with this is to put all chance "effects" into
the terminals: the terminal utilities returned will be already weighted by the
chance reach probabilities. Cf. values of all players include chance term, so
this is fine, and we can easily propagate them up the tree in Bottom-Up
traversal.

However, these chance reaches will be small values as they are cumulative
products of chance probabilities - in HUNL they will be $< 10^-6$, making it
difficult for neural network approximations. Therefore we also provide a method
for normalizing these chance probabilities.

## Design decisions

The factored-observations games have a very rich graph structure, and to encode
it efficiently there is a number of design decisions that must be made.

This document is intended for future reference, when the authors will forget why
they did things in a certain way and not another :-)

### Guidelines

-   Make it simple, no fancy C++ (templates, macros, etc.). KISS.
-   Make sure it is interoperable with Python.
-   Minimize inheritance as it is very much prone to leaky abstractions.
-   However, maximize composability. We should think about how to make each
    component as composable as possible.
-   Make sure it is possible to do (efficient) MCTS on the public tree.
-   Make sure it is easy to make a limited lookahead to some depth.
-   Handle cyclic games with imperfect recall abstraction (i.e. Dark Chess).
-   Properly mark `const` whenever applicable.
-   Proper use of smart pointers and ownership of objects.

### Compatibility with other parts of the code

OpenSpiel can (and probably will) grow to a library where many people want to
try out many different things. This is great, but at the same time we should aim
to make sure that OpenSpiel compiles fast. PS-API should be then an optional
dependency, enabled by a compilation flag.

Compatibility with Python and especially numpy is achieved with the use of Eigen
types that have a proper memory layout. See `eigen/` directory for more details.

### Static typing of entangled parts of code

We use `down_cast` instead of templates.

TODO(author13): explain reasons about "template" propagation to algorithms.

Maybe possible advantage -- Eigen optimizations? Is Eigen that powerful?

### Imperfect recall abstractions

TODO(author13): explain the precise meaning of the imperfect recall abstractions.

Note that the public states ill-definition problem of EFGs [1] does not arise in
our implementation, because we use action-private observation histories to
denote information states.

### Neural network interfaces

The public state encodes a constant-sized tensor, and we provide a mask on which
privates are “active” within a public state. We use const-sizes because it makes
it easier to work with in the algorithms. To encode and decode inputs to the
network we use `NetworkIndex()` and `ReachProbsIndex()` functions of
`PrivateInformation`.

### Simultaneous-move games

We support simultaneous-move games.

### (De)serialization support

We want to support (de)serialization via human-readable strings, similarly to
the Base API.

We do not aim to protobuf as it has too many weird dependencies.

### Use of vectorized operations

We aim to use vectorized operations in C++ whenever possible with Eigen library.

### Game transformations

To create PS-API from Base API we have quite a lot of freedom how to do so.
There are two "axis" (written here as "x" and "y") on which we can do it:

x. How "fine" is the partition of the world tree. y. Which resulting public
states we group together under imperfect recall.

The finest public partition of the world tree is common knowledge partition [1].
All other partitions (along the "x" axis) are derived as union of the elements
of the partition (i.e. union of public states).

Imperfect recall (the "y" axis) then allows to group the public states which
have the same sub-games, resp. the same value functions. A sub-game is a forest:
a collection of world trees, each rooted in the public state.

Because of these two choices, we can use different game transformations which
result in public trees of various sizes. Basically, the smaller the public tree
we want as a result of the transformation, the more computation it may require
to produce it.

Here is a list of a few possible game transformations:

1.  The players are only allowed to perceive the flow of time. This the coarsest
    possible representation.
2.  A perfect-recall coarse transformation based on public observation
    histories.
3.  A perfect-recall common knowledge transformation - we can use the recursive
    definition from [2] over (augmented) information states to create the public
    states.
4.  An imperfect-recall coarse transformation - for the public states produced
    by 2) we also compare the sub-games (resp. value functions) for equality.
5.  An imperfect-recall common knowledge transformation - for the public states
    produced by 3) we also compare the sub-games (resp. value functions) for
    equality.

A "qualitative" placement of the possible transformations on a plot would look
something like this:

```
            ^
  imperfect |    4             5
            |
recall      |          2
            |
    perfect | 1                3
            +-------------------
               < coarse   fine >
                   partition
```

Note that these transformations are perfectly fine also for perfect information
games: this is because we have the notion of beliefs! So if you imagine you play
Chess with transformation 1), theoretically it should work fine because you can
encode your true position in the game by setting belief = 1 to the corresponding
state.

### Consistency checks

The PS-API and Base API are tightly coupled. We can use this to our advantage to
create automatic consistency tests to make sure the PS-API is properly
implemented.

To go from PS-API to Base API we can use `PublicState` methods. To go from Base
API to PS-API we do not add methods to `State`, as that would create cyclic
dependencies in the source code and make it more complicated to do conditional
builds. We can use game transforms to do that, and check that the transformed
games are indeed consistent as well.

### Public tree structure

The structure of the public tree at the terminals should follow the observations
in the game. This means we need to be able to call
`Public/PrivateObservationString` at terminal states as well.

### Equality operators between structures

Imperfect recall and public histories? We might need to compare AOH with
infosets.

### What PS-API does not support:

-   Games that are not 1-timeable [1].
-   Games that have beliefs with exponential sizes.
-   FOG extensions:
    -   Observations shared between a subset of players.
    -   Observation consisting of features - observations are however directly
        encoded as either strings or tensors.
    -   Stochastic observations.

### Bunch of thoughts / notes

-   Pointer to Game, not GameWithPublicStates, prevent nested unwrapping calls.

## Refuted design decisions

-   Structure to observations - in some games we could say where the
    observations come from, like they are a result of chance, or they are
    observation of other player's actions, or direct result of him acting.
    Currently there is no alg that would use this, and it would make the API
    more complicated.
-   Templated support for arbitrary top-down / bottom-up propagation of values:
    Some algorithms maybe will require propagation of different objects than
    beliefs and counter-factual values, and we could provide that by templated
    support for `PublicState<TopDownObject, BottomUpObject>`, with template
    specialization to our current use of `PublicState<Beliefs, CfValues>`. Then
    each game would provide their own implementations, like `class
    KuhnPublicState<Beliefs, CfValues> : public PublicState<Beliefs, CfValues>`

    Instead, when there will be another use-case of public states (maybe for
    EFCEs/EFCCEs with a variant of Continual Resolving?) we can add specific
    computation methods as needed, and add `GameWithPublicStateType`, similar to
    `GameType` in Base API.

## Contributing

We would very much welcome if someone would implement DeepStack! Please get in
touch if you intend to do so.

There exists an implementation in Lua for Leduc Poker:
https://github.com/lifrordi/DeepStack-Leduc

List of published algorithms that can take advantage of this API:

-   [DeepStack](https://arxiv.org/abs/1701.01724)
-   [Pluribus](https://www.cs.cmu.edu/~noamb/papers/19-Science-Superhuman.pdf)
-   [MCCR](https://arxiv.org/pdf/1812.07351.pdf)

# References

If you use public state API in your research, please cite the FOG paper using
the following BibTeX:

```
@article{kovavrik2019rethinking,
  title={Rethinking formal models of partially observable multiagent decision making},
  author={Kova{\v{r}}{\'\i}k, Vojt{\v{e}}ch and Schmid, Martin and Burch, Neil and Bowling, Michael and Lis{\`y}, Viliam},
  journal={arXiv preprint arXiv:1906.11110},
  year={2019}
}
```

[1]: https://arxiv.org/abs/1906.11110
