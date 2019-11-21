# GridMaze Environment
This environment enables you to create and learn to solve grid mazes.
Solving mazes is a nice problem to start with when learning the basics of reinforcement learning. 

## Introduction
The problem description of the GridMaze environment is that you have a Robot located on a starting cell in a grid maze world. 
By moving Up, Down, Left, or Right, the robot should figure out the best strategy to move from the start cell to the goal cell of the maze.
The robot gets a reward (positive value) or a cost (negative value) when moving to a new cell. The purpose of the game is for the robot to figure out how to move from the start to goal position and getting as much reward (or as little cost) as possible.

Each cell in the grid world have different functions:
* **SPACE**: The robot is free to enter this cell, but there is a at a certain cost for doing so (e.g. -1).
* **START**: This is where the robot starts, like SPACE it can also be entered (returned to), at a cost. A maze must only have one START cell.
* **GOAL**: The robot successfully solves the maze and typically gets a reward by walking to this cell. A maze can have more than one goal cell.
* **WALL**: This cell cannot be entered.
* **BOUNCE**: The robot can attempt to enter this cell, but doing so will make it bounce back to the cell it came from, and get the associated reward/cost of that cell.
* **HOLE**: The robot enters this cell at a typically big cost, and the game is over. In other words, it fails to reach the GOAL cell.

Let's look at an example:
![ExampleImage](https://github.com/deepmind/open_spiel/blob/master/swift/Examples/GridMaze/Images/GridMazeExample.png?raw=true)

Describing each location in the maze by [row,column]:
* Robot starts out in cell [1,1]. From this cell it can only move right or down, since a wall cell cannot be entered. Going right or down will cost -1.
* The goal cell is at [4,4]. Entering this cell gives the robot 10 in reward, and the robot has successfully solve the maze (the T menas terminal cell).
* Entering the hole in [2,2] will cost the robot -100 and it is a terminal state (game over). Since robot did not reach goal cell, it lost. Likewise for the hole at [3,4].
* Entering one of the space cells at position [2,3] and [3,2] each has a cost of -2.
* The space cell at [4,0] is a bit special. Since the cell at [4,5] is also space, it means the robot can go left in [4,0] and end up in [4,5]. Had [4,5] been a wall, going left from [4,0] would not have been a legal action.
* Cell [4,2] is a bounce cell. This means attempting to enter it from [3,2], [4,1], or [4,3] results in robot bouncing back to those cells. Cost will be -2 from [3,2] and -1 from [4,1] and [4,3].

Now what about the white arrows with 50%? This indicates that cell [2,4] has random behavior.
If the robot attempts to enter cell [2,4], it will with 50% probability end up in that cell, but it might also well end up in cell [3,4] with 50% probability.

That's it for an overview of the environment, let's look at how interact with the environment through code!

## Creating a GridMaze Environment
This is how you create the maze environment in the picture above
```
var mazeEnvExample = GridMaze(rowCount: 6, colCount: 6)
mazeEnvExample[1, 1] = START()
mazeEnvExample[2, 2] = HOLE(reward: -100)
mazeEnvExample[2, 3] = SPACE(reward: -2.0)
mazeEnvExample[3, 2] = SPACE(reward: -2.0)
mazeEnvExample[3, 4] = HOLE(reward: -100)
mazeEnvExample[4, 0] = SPACE(reward: -1.0)
mazeEnvExample[4, 2] = BOUNCE()
mazeEnvExample[4, 4] = GOAL(reward: 1.0)
mazeEnvExample[4, 5] = SPACE(reward: -1.0)
mazeEnvExample[2, 4].entryJumpProbabilities = [(.Relative(1, 0), 0.5), (.Welcome, 0.5)]
```
As you can see, access to the cells in the environment is possible using the ```[row,column]``` syntax (GridMaze has overloaded the subscript operator).
By default, when you create and environment the cells will be surrounded by WALLs, and the free space inside it will be SPACE with cost -1 (see below for more information on changing this default behavior).

This code is pretty straightforward, except the last statement regarding the ```entryJumpProbabilities``` that needs some more explanation.
For any grid cell, you can specify the jump probabilities as follows
```
mazeEnvExample[row,col].entryJumpProbabilities = [
   // This is a list of jump probabilities
   // Each element is a pair, the first element specifies the jump behavior and the second the probability of it happening
   
   (.Welcome, 0.25),        // With 0.25 probability, welcome the robot to this cell
   (.BounceBack, 0.25),     // With 0.25 probability, make the robot bounce back to cell it was standing on
   (.Relative(1,0), 0.25),  // With 0.25 probability, jump to the next row (1), same column (0) in the matrix
   (.Absolute(1,1), 0.25),  // With 0.25 probability, jumpt to cell at row 1, column 1  
]
```
You can also omit specifying the ```entryJumpProbabilities```, in case the default for the cell type will apply, i.e.
* SPACE, START, GOAL, and HOLE will be ```[(.Welcome, 1.0)]```
* BOUNCE will be ```[(.BounceBack), 1.0]```

A couple of rules related to entryJumpProbabilities
* If you do specify this field, it must be either have 0 elements (== 100% ```.Welcome```), or probabilities in list must sum to 1.0.
* When using ```.Relative``` or ```.Absolute```, the target cell must be 100% ```.Welcome```. The environment current does not support sequenced jumps.

There is also currently limilted environment validation (contributions are welcome).
In other words, make sure you index cells that exist, and if you're doing ```entryJumpProbabilities```, that any ```.Relative``` or ```.Absolute``` target cell exist (and does not in turn have ```entryJumpProbabilities```).

The ```init``` function of GridMaze looks like this
```
public init(
  rowCount: Int,
  colCount: Int,
  cellsLeftSide: GridCell = WALL(),
  cellsTopSide: GridCell = WALL(),
  cellsBottomSide: GridCell = WALL(),
  cellsRightSide: GridCell = WALL(),
  cellsTopLeft: GridCell = WALL(),
  cellsTopRight: GridCell = WALL(),
  cellsBottomLeft: GridCell = WALL(),
  cellsBottomRight: GridCell = WALL(),
  cellsAllOthers: GridCell = SPACE(reward: -1.0)
```
As you can see, the size of the environment must be specified through the rowCount and colCount paramaters.
Using default parameter values, the environment will be surrounded by WALLs, and the other cells will be SPACE with a cost of -1.0.
You can change the default by specifying these parameters, for example surround it with **BOUNCE** by providing BOUNCE() instead of the default **WALL()**.

## Printing a Maze
After a maze has been created, it can be printed by calling printMazeAndTable
```
mazeEnvExample.printMazeAndTable(header: "Maze Environment")
```

This gives the following output:
```
Maze Environment

Transition probabilities (non-stochastic transitions are not printed (i.e as expected based on action)
[2,4]: Probability: 0.50, Type: Relative (1,0)
       Probability: 0.50, Type: Welcome
[4,2]: Probability: 1.00, Type: BounceBack

             00            01            02            03            04            05       
00          WALL          WALL          WALL          WALL          WALL          WALL      

01          WALL       START:-1:I     SPACE:-1      SPACE:-1      SPACE:-1        WALL      

02          WALL        SPACE:-1     HOLE:-100:T    SPACE:-2    SPACE(JS):-1      WALL      

03          WALL        SPACE:-1      SPACE:-2      SPACE:-1     HOLE:-100:T      WALL      

04        SPACE:-1      SPACE:-1     BOUNCE(JS)     SPACE:-1      GOAL:1:T      SPACE:-1    

05          WALL          WALL          WALL          WALL          WALL          WALL      
```

This function prints:
* Maze with all cells and their rewards (WALL and BOUNCE do not have rewards, since they cannot be entered).
* Letter 'T' indicates that the cell is terminal
* A cell that contains '(JS)' means it has a JumpSpecification, which is detailed before the maze printout.

### Printing V-/Q-Values and Policy
(if you are not familiar with these concepts, then please see the [OpenSpiel Swift tutorials](https://github.com/tensorflow/examples/blob/master/community/en/swift/open_spiel)).

It is also possible to print a v-/q-/policy-table together with the maze.

In the below code:
* **vtable** is a mapping from state to the value in that state
* **qtable** is a mapping from state to a value for each of the legal actions from that state
* **ptable** is a mapping from state to the action the policy takes from that state
```
var vtable = [String: Double]()
var qtable = [String: [(action: GridMaze.Action, qvalue: Double)]]()
var policyTable = [String: [GridMaze.Action]]()
mazeEnvExample.printMazeAndTable(header: "--- Printing Everything", vtable: vtable, qtable: qtable, ptable: policyTable)
```
Since these tables do not contain any data for the cells
* **vtable** will print the value 0 for each state
* **qtable** will print a uniform distribution across the legal actions from each cell
* **ptable** will print '?' since there is no policy for any of the states

## Creating Custom Cells
The cell types we've looked at so far (START, GOAL, SPACE, HOLE, BOUNCE) are all just functions that GridCell objects with certain parameters.
You can create your own cell behaviors by calling GridCell.init with the desired behavior.
See the GridMaze.swift source code to understand how this works.

## Putting it all together
The file ```open_spiel/swift/Examples/GridMaze/main.swift``` is a complete example of a Robot walking randomly through the maze trying to reach the goal cell.
The [OpenSpiel Swift tutorials](https://github.com/tensorflow/examples/blob/master/community/en/swift/open_spiel) also  have more advanced examples, where the robot learns the optimal ways to navigate through the maze.

## Join the community!
If you have any questions about Swift for TensorFlow, Swift in OpenSpiel, or would like to share
your work or research with the community, please join our mailing list
[`swift@tensorflow.org`](https://groups.google.com/a/tensorflow.org/forum/#!forum/swift).
