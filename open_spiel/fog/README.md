# Factored-Observation Games

This directory contains code implementing Factored-Observation Games (FOGs) from
https:arxiv.org/abs/1906.11110 [1].

FOGs are games where players can distinguish between so-called public and
private observations. The implementation tries to follow the formalism in [1]
closely, but there are some important differences, which are listed below.

The public observations correspond to the information that all the players know
that all the players know, like upward-facing cards on a table or passing of
time (clock ticking). The private observation is then the remaining information,
distinct from the public observation. Thus private and public observations are
non-overlaping. Their union corresponds to the (undistinguished) player
observation available as `State::ObservationString()`. The union of (public and
private informantion) and `ObservationString` should partition the game tree
identically (even though this is not checked).

The public / private observations can be accessed by
`State::PublicObservationString()` and by
`State::PrivateObservationString(Player)`. Currently Tensor versions are not
provided.

There are some subtle, but important differences from FOGs:

-   In FOGs, there is a concept of a "world state", which represents the true
    current state of the world from a perspective of an omniscient being (that
    knows everything). The players may not be certain which world state they are
    currently in, as they only receive observations in the game, and they have
    to deduce what world states are compatible with these observations.

    In OpenSpiel, the `State` tracks the world state AND the (omniscient)
    history of actions of all players. Action history is an alternative way to
    define the `State`, **but it may not be unique**!

    Consider following: Let x and y be some world states. If
    action_history(x) == action_history(y) then x == y for all games. However,
    there can exist x and y such that action_history(x) != action_history(y) but
    x == y.

    In other words, there may exist distinct action histories that represent the
    same world state. For example games on cyclic graphs (security/pursuit-
    evasion games), where the agents revisit the same position in the graph.

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
    observations: this includes chance `State`.

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
    observations). Both of these ways of representation have a string and tensor
    representation, and they partition the game tree in the same way.

The factored observations can be used to deduce observations (as concatenation
of private and public observations). `ActionObservationHistory` can be used to
deduce information states. Therefore factored observations are the most general
formalism to describe imperfect information games supported by OpenSpiel. They
are also used for automatic generation of public state API via game
transformations, see `public_states/README.md` for more details.
