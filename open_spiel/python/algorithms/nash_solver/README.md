# Nash Solver for PSRO
This directory provides a general Nash equilibrium solver that can solve general-sum many-player games. The main entry is the `nash_solver` function in `general_nash_solver.py`. This solver is designed for PSRO algorithm with meta games as input and a list of equilibria as output.
## Arguments
* `meta_games`: meta_games (empirical payoff matrix) object in PSRO.
* `solver`: string name of a Nash equilibrium solver.
* `mode`: choose whether return all equilibria or pure equilibria or one equilibrium.
* `lrsnash_path`: path to the lrsnash solver if `solver == 'lrsnash'`. Default value is `None` and the system searches the solver automatically.


## Solver Options
* **'gambit'**: Gambit is a library of game theory software and tools for the construction and analysis of finite extensive and strategic games. For more information see:[http://www.gambit-project.org/gambit16/16.0.0/index.html] (http://www.gambit-project.org/gambit16/16.0.0/index.html). Gambit is the only option for solving games with more than 2 players. 
* **'replicator'**: Find an equilibrium using replicator dynamics.
* **'lrsnash'**: lrsnash uses reverse search vertex enumeration on rational polytopes. For more info see: [http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash](http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html#nash).
* **'nashpy'**: nashpy is a Python library used for the computation of equilibria in 2 player strategic form games. For more info see: [https://nashpy.readthedocs.io/en/latest/](https://nashpy.readthedocs.io/en/latest/).
* **'linear'**: solve a matrix game by using linear programming. This method only works for two-player zero-sum games. 

## Mode Options
* **'all'**: return all equilibria.
* **'pure'**: return pure equilibria.
* **'one'**: return the first equilibrium. 

## Requirements
Before `gambit` or `Irsnash` option is used, please install the corresponding backend. Please refer to their websites for installation details.  

## Tips
* For solving large games, please select "gambit" or "replicator" as the solver. Other solvers could be hanging.
* To find all equilibria, different methods in "gambit" may give different answers. Some methods may not be able to find all the equilibria. Please refer to the gambit document and change the method if necessary. The default method is "gnm".
* "replicator" solver only returns one equilibrium depending on the initial point.
