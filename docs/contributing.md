# Guidelines

Above all, OpenSpiel is designed to be easy to install and use, easy to
understand, easy to extend (“hackable”), and general/broad. OpenSpiel is built
around two major important design criteria:

-   **Keep it simple.** Simple choices are preferred to more complex ones. The
    code should be readable, usable, extendable by non-experts in the
    programming language(s), and especially to researchers from potentially
    different fields. OpenSpiel provides reference implementations that are used
    to learn from and prototype with, rather than fully-optimized /
    high-performance code that would require additional assumptions (narrowing
    the scope / breadth) or advanced (or lower-level) language features.

-   **Keep it light.** Dependencies can be problematic for long-term
    compatibility, maintenance, and ease-of- use. Unless there is strong
    justification, we tend to avoid introducing dependencies to keep things easy
    to install and more portable.

# Support expectations

We, the OpenSpiel authors, definitely engage in supporting the community. As it
can be time-consuming, we try to find a good balance between ensuring we are
responsive and being able to continue to do our day-to-day work and research.

Generally speaking, if you are willing to get a specific feature implemented,
the most effective way is to implement it and send a Pull Request. For large
changes, or ones involving design decisions, open a bug to check the idea is ok
first.

The higher the quality, the easier it will be to be accepted. For instance,
following the
[C++ Google style guide](https://google.github.io/styleguide/cppguide.html) and
[Python Google style guide](http://google.github.io/styleguide/pyguide.html)
will help with the integration.

As examples, MacOS support, Window support, example improvements, various
bug-fixes or new games has been straightforward to be included and we are very
thankful to everyone who helped.

## Bugs

We aim to answer bugs at a reasonable pace, several times a week. However, for
bugs involving large changes (e.g. adding new games, adding public state
supports) we cannot commit to implementing it and encourage everyone to
contribute directly.

## Pull requests

You can expect us to answer/comment back and you will know from the comment if
it will be merged as is or if it will need additional work.

For pull requests, they are merged as batches to be more efficient, at least
every two weeks (for bug fixes, it will likely be faster to be integrated). So
you may need to wait a little after it has been approved to actually see it
merged.

# Roadmap and Call for Contributions

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). See
[CONTRIBUTING.md](https://github.com/deepmind/open_spiel/blob/master/CONTRIBUTING.md)
for the details.

Here, we outline our intentions for the future, giving an overview of what we
hope to add over the coming years. We also suggest a number of contributions
that we would like to see, but have not had the time to add ourselves.

Before making a contribution to OpenSpiel, please read the guidelines. We also
kindly request that you contact us before writing any large piece of code, in
case (a) we are already working on it and/or (b) it's something we have already
considered and may have some design advice on its implementation. Please also
note that some games may have copyrights which might require legal approval.
Otherwise, happy hacking!

The following list is both a Call for Contributions and an idealized road map.
We certainly are planning to add some of these ourselves (and, in some cases
already have implementations that were just not tested well enough to make the
release!). Contributions are certainly not limited to these suggestions!

-   **Checkers / Draughts**. This is a classic game and an important one in the
    history of game AI
    (["Checkers is solved"](https://science.sciencemag.org/content/317/5844/1518)).

-   **Chinese Checkers / Halma**.
    [Chinese Checkers](https://en.wikipedia.org/wiki/Chinese_checkers) is the
    canonical multiplayer (more than two player) perfect information game.
    Currently, OpenSpiel does not contain any games in this category.

-   **Deep TreeStrap**. An implementation of TreeStrap (see
    [Bootstrapping from Game Tree Search](https://www.cse.unsw.edu.au/~blair/pubs/2009VenessSilverUtherBlairNIPS.pdf)),
    except with a DQN-like replay buffer, storing value targets obtained from
    minimax searches. We have an initial implementation, but it is not yet ready
    for release. We also hope to support PyTorch for this algorithm as well.

-   **Deep Regret Minimization with Advantage Baselines and Model-free Learning
    (DREAM)**. This is a model-free technique based on Monte Carlo CFR with
    function approximation, that has been applied to Poker.
    ([Ref](https://arxiv.org/abs/2006.10410))

-   **Double Neural Counterfactual Regret Minimization**. This is a technique
    similar to Regression CFR that uses a robust sampling technique and a new
    network architecture that predicts both the cumulative regret *and* the
    average strategy. ([Ref](https://arxiv.org/abs/1812.10607))

-   **Differentiable Games and Algorithms**. For example, Symplectic Gradient
    Adjustment ([Ref](https://arxiv.org/abs/1802.05642)).

-   **Emergent Communication Algorithms**. For example,
    [RIAL and/or DIAL](https://arxiv.org/abs/1605.06676) and
    [CommNet](https://arxiv.org/abs/1605.07736).

-   **Emergent Communication Games**. Referential games such as the ones in
    [Ref1](https://arxiv.org/abs/1612.07182),
    [Ref2](https://arxiv.org/abs/1710.06922),
    [Ref3](https://arxiv.org/abs/1705.11192).

-   **Extensive-form Evolutionary Dynamics**. There have been a number of
    different evolutionary dynamics suggested for the sequential games, such as
    state-coupled replicator dynamics
    ([Ref](https://dl.acm.org/citation.cfm?id=1558120)), sequence-form
    replicator dynamics ([Ref1](https://arxiv.org/abs/1304.1456),
    [Ref2](http://mlanctot.info/files/papers/aamas14sfrd-cfr-kuhn.pdf)),
    sequence-form Q-learning
    ([Ref](https://dl.acm.org/citation.cfm?id=2892753.2892835)), and the logit
    dynamics ([Ref](https://dl.acm.org/citation.cfm?id=3015889)).

-   **General Games Wrapper**. There are several general game engine languages
    and databases of general games that currently exist, for example within the
    [general game-playing project](http://www.ggp.org/) and the
    [Ludii General Game System](http://www.ludii.games/index.html). A very nice
    addition to OpenSpiel would be a game that interprets games represented in
    these languages and presents them as OpenSpiel games. This could lead to the
    potential of evaluating learning agents on hundreds to thousands of games.

-   **Go API**. We currently have an experimental [Go](https://golang.org/) API
    similar to the Python API. It is exposed using cgo via a C API much like the
    CFFI Python bindings from the
    [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment).
    It is very basic, only exposing the games. It would be nice to have a few
    example algorithms and/or utilities written in go.

-   **Minimax-Q and other classic MARL algorithms**. Minimax-Q is a classic
    multiagent reinforcement learning algorithm
    ([Markov games as a framework for multi-agent reinforcement learning](https://www2.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf).
    Other classic algorithms, such as
    [Correlated Q-learning](https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf),
    [NashQ](http://www.jmlr.org/papers/volume4/hu03a/hu03a.pdf), and
    Friend-or-Foe Q-learning
    ([Friend-or-foe q-learning in general-sum games](http://jmvidal.cse.sc.edu/library/littman01a.pdf)
    would be welcome as well.

-   **Nash Averaging**. An evaluation tool first described in
    [Re-evaluating Evaluation](https://arxiv.org/abs/1806.02643).

-   **Opponent Modeling / Shaping Algorithms**. For example,
    [DRON](https://arxiv.org/abs/1609.05559),
    [LOLA](https://arxiv.org/abs/1709.04326), and
    [Stable Opponent Shaping](https://arxiv.org/abs/1811.08469).

-   **Rust API**. We currently have an experimental
    [Rust](https://www.rust-lang.org/) API. It is exposed via a C API much like
    the Go API. It is very basic, only exposing the games. It would be nice to
    have a few example algorithms and/or utilities written in Rust.

-   **Sequential Social Dilemmas**. Sequential social dilemmas, such as the ones
    found in [Ref1](https://arxiv.org/abs/1702.03037),
    [Ref2](https://arxiv.org/abs/1707.06600) . Wolfpack could be a nice one,
    since pursuit-evasion games have been common in the literature
    ([Ref](http://web.media.mit.edu/~cynthiab/Readings/tan-MAS-reinfLearn.pdf)).
    Also the coin games from [Ref1](https://arxiv.org/abs/1707.01068) and
    [Ref2](https://arxiv.org/abs/1709.04326), and Clamity, Cleanup and/or
    Harvest from [Ref3](https://arxiv.org/abs/1812.07019)
    [Ref4](https://arxiv.org/abs/1810.08647).

-   **Structured Action Spaces**. Currently, actions are integers between 0 and
    some value. There is no easy way to interpret what each action means in a
    game-specific way. Nor is there any way to easily represent a composite
    action in terms of its parts. A structured action space could represent
    actions as a sequence of values (like information states and observations--
    and can also include shapes) which can be learned instead of mappings to
    flat numbers. Then, each game could have a mapping from the structured
    action to the action taken.

-   **TF_Trajectories**. The source code currently includes a batch inference
    for running a batch of episodes using Tensorflow directly from C++ (in
    `contrib/`). It has not yet been tested with CMake and public Tensorflow. We
    would like to officially support this and move it into the core library.

-   **Visualizations of games**. There exists an interactive viewer for
    OpenSpiel games called [SpielViz](https://github.com/michalsustr/spielviz).
    Contributions to this project, and more visualization tools with OpenSpiel,
    are welcome.

-   **Windows support**. Native Windows support was added in early 2022, but
    remains experimental and only via building from source. It would be nice to
    have Github Actions CI support on Windows to ensure that Windows support is
    actively maintained, and eventually support installing OpenSpiel via pip on
    Windows as well.
