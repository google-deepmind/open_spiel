# Factored-Observation Games

This directory contains code implementing Factored-Observation Games (FOGs) from
https:arxiv.org/abs/1906.11110 [1].

FOGs are games where players can distinguish between so-called public and
private observations. The implementation tries to follow the formalism in [1]
closely, but there are some important differences, which are listed below.

The public observations correspond to the information that all the players know
that all the players know, like upward-facing cards on a table or passing of
time (clock ticking). The private observation is then the remaining information,
distinct from the public observation, with the exception of time for technical
reasons. Thus private and public observations are almost non-overlaping.

The respective observations with (im)perfect recall can be accessed via the
`Observer` tensor / string methods.

There are some subtle, but important differences from FOGs:

-   In FOGs players take joint actions, and each player has at least one action.
    In OpenSpiel there is a version of simultaneous-move games which does this.
    The (more common) sequential game **can be interpreted as a
    simultaneous-move game**, as all the other players just played a "no-op"
    action.

-   After the joint actions are taken, in FOGs there is always a transition
    function to new world states, which encodes stochasticity (and is
    deterministic if there is only one outcome). In OpenSpiel, this is done via
    **explicit chance** `State`.

-   In FOGs the players receive observations only after the stochastic
    transition. In OpenSpiel **we can ask at any `State`** for the newest
    observations: this includes any chance `State`.

-   In FOGs the "world history" is encoded as a sequence of tuples (world, joint
    actions), because the stochastic transition can put the players into
    different world states. In OpenSpiel, the action history is equivalent,
    because chance player also contributes its action to the action history
    (this means we don't need to keep track of world states).

-   In FOGs, the imperfect information of a player is encoded by using
    `ActionObservationHistory` histories (AOH), where "Observation" here
    represents the joint private and public observations. In OpenSpiel, we
    implement a variant of AOH which works also for the common sequential game.
    Besides AOH there is also an "information state", which is a non-factored
    variant of AOH (it is not factored into the respective actions and
    observations). Even though information state is non-factored, it is perfect
    recall in the sense that this factorization should be possible in a
    domain-specific way. AOH is a general implementation of this factorization.
    Both of these ways of representation have a string and tensor
    representations, and they partition the game tree in the same way.

The factored observations can be used to deduce observations (as concatenation
of private and public observations). `ActionObservationHistory` can be used to
deduce information states. Therefore factored observations are the most general
formalism to describe imperfect information games supported by OpenSpiel. They
are also used for automatic generation of public state API via game
transformations, see `public_states/README.md` for more details.

The game-specific information state string will in general be shorter and
possibly more human-readable. The game-specific information state tensor will
usually be smaller and should also have been designed to make learning and
generalization easier for a neural net.
