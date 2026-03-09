# Mean Field Games Open Spiel Python API

This is a Python API for Mean Field Games in OpenSpiel.

> [!WARNING] This code is experimental and we recommend you not to use it yet
> unless you are part of the project.

> [!NOTE] Checkout the C++ API `../../games/mfg/README.md` documentation

## What am I doing here?

This API targets algorithms and games modelling interactions of agents in very
large populations (put simply, multi-agent systems with number of agents
approaching a large number, $N\to\infty$). According to
[wikipedia](https://en.wikipedia.org/wiki/Mean-field_game_theory), the use of
the term "mean field" is inspired by mean-field theory in physics, which
considers the behavior of systems of large numbers of particles where individual
particles have negligible impacts upon the system.

If you look forward to solving problems with a finite number of interacting
agents, please, refer to the main `OpenSpiel` API. You can find out more
examples of the api usage in the corresponding [`examples`](./examples/)
directory.

## Implemented games

### Crowd avoidance game

A classic MFG that was derived from broad analisys of "Mean Field Games for
Pedestrian Dynamics" (https://arxiv.org/abs/1304.5201).

*   *State*: The agent’s position on a 1D or 2D grid and the current time step.
*   *Actions*: Moving to adjacent cells.
*   *Reward*: A combination of a terminal reward (reaching a goal) and an
    instantaneous cost that is proportional to the density of other agents in
    the same cell.

Instead of tracking $N$ individual players, the game tracks a probability
distribution (i.e., the "population state"). Each agent optimises its path based
on the assumption that the rest of the crowd will also move to minimise their
own congestion.

The environment is configurable in the following high-level ways: - Congestion
coefficients matrix. - Initial distribution. - Geometry (torus, basic square).

### Crowd modelling game (2D)

Mean Field Crowd Modelling Game (in 2D). This game corresponds to a 2D "Beach
Bar Process" defined in section 4.2 of "Fictitious play for mean field games:
Continuous time analysis and applications" (https://arxiv.org/abs/2007.03458).

*   *State*: the game takes place on a $N \times N$ grid, which includes
    forbidden states: certain cells (marked as '#' in the code) are "walls" or
    obstacles. Agents cannot enter these cells. Initial Distribution: The agents
    start at a specific location or spread across the grid.

*   *Actions*: Up, Down, Left, Right, or choose to Stay in their current cell
    (analogously, Stay, Left, and Right in the 1D case).

If an agent attempts to move into a "forbidden state" (a wall) or outside the
grid boundary, the move is blocked, and they remain in their current cell.

The movement is deterministic at the agent level, but since this is a Mean Field
Game, we track how the entire population density shifts across the grid at each
time step.

*   *Rewards*: The "game" part comes from the cost function, which usually
    consists of three components: `Congestion Cost` + `Movement Cost` + `Goal
    Reward`.

Every individual agent is choosing a path that minimises their own cost (given
where the crowd is).

For 2D, the environments "MAZES" (100 max steps) and "FOUR_ROOMS" (40 max steps)
are pre-implemented.

### Dynamic routing

The Dynamic Routing MFG simulates a traffic flow where a "representative"
vehicle navigates a road network, where a population of drivers is represented
as a continuous density.

The game is based on the paper "Deep Reinforcement Learning for General-Sum Mean
Field Games of Dynamic Steering" (https://arxiv.org/abs/2110.11943).

*   *State*: a vehicle's state is defined by its `(current_location,
    current_waiting_time, destination)`. And the road network is modelled as
    graph.

*   *Actions*: num links of the graph nodes (can be equivalent to the spatial
    actions in the "Crowd modelling game.")

*   *Reward*: Utility = Arrival Time: The "utility" (score) for a vehicle is
    typically the negative of its total travel time (or arrival time). Because
    everyone wants the fastest route, they naturally compete for the same roads,
    leading to the congestion that the MFG algorithms try to solve.

The objective for each "representative vehicle" is to reach their destination as
quickly as possible when traffic depends on how many other people are using that
road at the same time (congestion).

### Linear quadratic (LQ) game

While games like "Crowd Modelling" or "Dynamic Routing" are specific to traffic
or pedestrians, the LQ game is a general-purpose template used across economics,
finance (portfolio optimisation), and engineering. This game is derived from
"Fictitious play for mean field games: Continuous time analysis and
applications"(https://arxiv.org/abs/2007.03458). This game corresponds to the
game in section 4.1.

*   *State* ($x$): a continuous value (e.g., your location on a 1D line).
*   *Action* ($u$): The "push" or "acceleration" that is applied to advance the
    state according to the LQR dynamics.
*   *Reward/Cost*: Reward maximisation is defined as the cost minimisation. if

    *   $\Delta t$ = the time step size (`dt`)
    *   $\bar{x} - x$ = the distance from the mean (`dist_mean`)
    *   $q$ = the cross-coupling coefficient (`cross_q`)
    *   $\kappa$ = the congestion or aversion parameter (`kappa`), then, the
        running reward $R$ is: $$R(x, u) = \frac{\Delta t}{2} \left( -u^2 + 2q
        u(\bar{x} - x) - \kappa(\bar{x} - x)^2 \right).$$

    At the end of the game, the agent receives an additional penalty based on
    its final position: $$R_{terminal} = -\frac{\gamma}{2}(\bar{x} - x)^2,$$
    where $\gamma$ is controlled by `terminal_cost`. This forces the agent to
    finish the simulation as close to the population mean as possible.

The goal of the game is to find a policy where, if every single population
member follows it, the resulting average population state is exactly what
everyone expected it to be.

### Normal form game

This is a MFG form of a classic "Normal Form" (one-shot) game.

*   *Reward*: The agent receives a reward based on a pre-defined mathematical
    formula using that distribution.

### Periodic aversion

The game was introduced to study ergodic MFG with explicit solution in Almulla,
N.; Ferreira, R.; and Gomes, D. 2017 of "Two numerical approaches to stationary
mean-field games". Dyn. Games Appl. 7(4):657-682
(https://arxiv.org/abs/1511.06576).

The environment is a 1D line of a certain size (e.g., 0 to 10), but it is
periodic. This means if an agent moves past the maximum value, they "wrap
around" to the beginning.

*   *State*: each agent is defined by their `(position, time_step)`.

*   *Actions*: at each time step, the agent chooses a movement, modelled by a
    discrete set of discrete navigation actions (e.g., move left, stay, or move
    right).

*   *Reward* (here, negative cost) is composed of the following factors:
    `Congestion Cost` + `Movement Cost` + `Goal Distance`. If the agent is
    alone, this penalty is zero, but if the agent is in a "crowd," the penalty
    increases by a predefined function.

The game tracks the distribution of all agents across the circle at every time
step. It could be useful to study circular flows and "traffic waves" that don't
have a starting or ending point.

### Predator-prey game

This game corresponds to the predator-prey game described in section 5.4 of
"Scaling up Mean Field Games with Online Mirror Descent"
(https://arxiv.org/abs/2103.00623)

*   *State*: positions and targets of both agents (predator and prey) on the 2D
    grid.
*   *Actions*: both populations can move in the standard grid directions (Up,
    Down, Left, Right, Stay).
*   *Rewards* are "coupled" between the two populations:
    1.  Predator Reward ($R_{pred}$): $R = \text{Goal distance} - \eta *
        \log(\mu(x))$ where $\mu$ is a stationary distribution,
    2.  Prey Reward ($R_{prey}$) which is defined likewise,
    3.  A Congestion Penalty.

The environment is configurable in the following high-level ways: - Number of
populations. - Reward matrix. - Initial distribution. - Geometry (torus, basic
square).

### Garnet Game

> [!NOTE] This game is only implemented in C++, see
> `../../games/mfg/garnet.cc`. `Python` contributions are welcome

This game corresponds to a garnet defined in section 5.1 of "Scaling up Mean
Field Games with Online Mirror Descent" (https://arxiv.org/abs/2103.00623.pdf)

Quoting the paper, Garnet is a parametrized family of randomly generated static
Mean Field Games. The reward is parametrized by $\eta$ as $r(x,a) - \eta *
\log(\mu(x))$ where $r(x,a)$ is sampled uniformly over $[0,1]$ only with
probability the sparsity and $0.0$ otherwise.

The environment is configurable in the following high-level ways: - Number of
actions. - Reward matrix and its sparsity. - Number of chance actions.

## Additional readings

-   "Bench-MFG: A Benchmark Suite for Learning in Stationary Mean Field Game"
    (https://arxiv.org/abs/2602.12517)
