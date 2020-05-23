// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// This file implements an example program using the OpenSpiel library

import OpenSpiel
import TensorFlow

// Create the GridMaze environment the Robot should learn to solve
var mazeEnv = GridMaze(rowCount: 6, columnCount: 6)
mazeEnv[1, 1] = GridCell.start(reward: -1.0)
mazeEnv[2, 2] = GridCell.hole(reward: -100)
mazeEnv[2, 3] = GridCell.space(reward: -2.0)
mazeEnv[3, 2] = GridCell.space(reward: -2.0)
mazeEnv[3, 4] = GridCell.hole(reward: -100)
mazeEnv[4, 0] = GridCell.space(reward: -1.0)
mazeEnv[4, 2] = GridCell.bounce
mazeEnv[4, 4] = GridCell.goal(reward: 1.0)
mazeEnv[4, 5] = GridCell.space(reward: -1.0)
mazeEnv[2, 4].entryJumpProbabilities = [(.Relative(1, 0), 0.5), (.Welcome, 0.5)]

// Print environment
mazeEnv.printMazeAndTable(header: "Maze Environment")

var gamesPlayed = 0
while true {
  gamesPlayed += 1
  var gameState = mazeEnv.initialState
  var actionSequence = [String]()
  
  // Let the Robot take actions (walk betwen cells) until it either reaches GOAL or
  // falls into a HOAL (those are the only two terminal cells)
  while !gameState.isTerminal {
    
    // The current informationState, denoting the cell position the Robot is currently at
    // If you create a learning algorithm, this will be the key to v-/q-/policy-tables
    // let currentInformationState = gameState.informationStateString()
    
    // Select a random action from the legal ones in this state
    // The GridMaze.Action enum has members: .LEFT, .UP, .DOWN, .RIGHT
    // Since some cells cannot be even attempted to be entered (e.g. WALL), all gameStates may not return all four members
    let actionIndex = Int.random(in: 0..<gameState.legalActions.count)
    let actionToTake = gameState.legalActions[actionIndex]
    
    // We now have an actionToTake from current informationState
    // Let's have Robot move in that direction!
    gameState.apply(actionToTake)
    
    // Store the action taken to print that later
    actionSequence.append(actionToTake.description)
    
    // If Robot made it to GOAL then we're done
    if gameState.isGoal {
      print("*** AWESOME. Robot made it to Goal at game: \(gamesPlayed) with reward/cost: \(gameState.utility(for: .player(0)))")
      print("    Sequence of actions used to solve the maze: \(actionSequence)")
      exit(0)
    } else if gameState.isTerminal {
      print("Robot failed to solve the maze at game: \(gamesPlayed) with reward/cost: \(gameState.utility(for: .player(0)))")
    }
  }
}
