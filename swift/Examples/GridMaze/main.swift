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

// We are assuming the reader knows what QValues are (TODO:)

// Let's say we have a Robot that wants to learn how to navigate from the START grid cell
// to the GOAL grid cell with the highest reward score possible.

// Create the GridMaze environment the Robot should learn to solve
var mazeEnv = GridMaze(rowCount: 7, colCount: 7, cellBottomSide: WALL())
mazeEnv[5, 1] = START()
mazeEnv[5, 2] = HOLE(reward: -100)
mazeEnv[5, 3] = HOLE(reward: -100)
mazeEnv[5, 4] = HOLE(reward: -100)
mazeEnv[5, 5] = GOAL(reward: 1.0)
mazeEnv[4, 2].entryJumpProbabilities = [(.Relative(1, 0), 0.5), (.Welcome, 0.5)]
mazeEnv[4, 3].entryJumpProbabilities = [(.Relative(1, 0), 0.5), (.Welcome, 0.5)]
mazeEnv[4, 4].entryJumpProbabilities = [(.Relative(1, 0), 0.5), (.Welcome, 0.5)]

// Print the maze
// If the Robot enters the grid cells (SPACE) immediately above the HOLE cells,
// there is a 50% probability of dropping down into the HOLE cell and a
// 50% probability of entering those SPACE cells
mazeEnv.printMazeAndTable(header: "Maze Environment")

// Create the QTable which is a mapping from a specific information state
// (i.e. specific GridCell in the maze) to a list of actions possible from that
// cell and their associated qvalue. When the Robot is fully trained, it will
// select the action with the highest qvalue from any given cell.
var qtable = [String: [(action: GridMaze.Action, qvalue: Double)]]()

// Print the Maze with the qtable. Since the QValues
mazeEnv.printMazeAndTable(header: "--- Before any iteration", qtable: qtable)

// Let's just check what an informationState is.
// initialState returns "5:1", which is the start cell.
// This is just for demonstrational purposes, you should never rely on the
// format/implementation details of informationState().
let gameState = mazeEnv.initialState
print("Printing starting informationState: \(gameState.informationState())")

// Ok, let's now train the Robot by letting it walk around the maze and learn.
// For 100,000 times, let it start from START cell and try to get to GOAL position
// as effectively as possible (i.e. maximizing the reward).
var episodeCount = 0
while episodeCount < 100000 {
  episodeCount += 1

  // Start a episode, position robot at START.
  // Observe that gameState is something different from informationState
  var gameState = mazeEnv.initialState

  var totalReward = 0.0  // Total reward collected so far

  // Let the Robot take actions (walk betwen cells) until it either reaches GOAL or
  // falls into a HOAL (those are the only two terminal cells)
  while !gameState.isTerminal {

    // The current informationState, denoting the cell position the Robot is currently at
    let currentInformationState = gameState.informationState()

    // If state has never been seen, create qvalues (0.0) for all legal actions from it
    if !qtable.keys.contains(currentInformationState) {
      qtable[gameState.informationState()] = gameState.legalActions.map { ($0, 0.0) }
    }

    // Should the Robot explore new possible paths, or follow the ones it (qvalue) so far
    // has determined is the most optimal.
    // With 0.9 probability, select action with max Q value in state
    //      0.1 probability, select a random action (ensures exploration)
    let isGreedy = Double.random(in: 0.0..<1.0) < 0.9

    // This section populates actionToTake from the current informationState
    var actionToTake: GridMaze.Action? = nil
    if isGreedy {  // If acting greedily
      // As per above, qvalues for informationState must exist
      // Select the maximum qvalue from this informationState
      let actionValue = qtable[gameState.informationState()]!.max(by: {
        // If two elements are equal, then select one by random
        if $0.qvalue == $1.qvalue {
          return Double.random(in: 0.0..<1.0) < 0.5
        }
        return $0.qvalue < $1.qvalue  // Not equal, select the maximum value
      })
      actionToTake = actionValue!.action
    } else {  // Not greedy, explore
      // Select a random action from the legal set
      // (could be even smarter and avoid selecting the action with the highest qvalue
      // but not doing this for code simplicitly)
      let actionValueCount = qtable[gameState.informationState()]!.count
      actionToTake
        = qtable[gameState.informationState()]![Int.random(in: 0..<actionValueCount)].action
    }

    // We now have an actionToTake from current informationState
    // Let's have Robot move in that direction!
    gameState.apply(actionToTake!)

    // Get the new informationState and the reward given (utility returns reward since start of game)
    let newInformationState = gameState.informationState()
    let newTotalReward = gameState.utility(for: .player(0))  // TODO: Would be great if argument defaulted to .player(0)
    let actionReward = newTotalReward - totalReward
    totalReward = newTotalReward

    // Update the qtable based on the informationState we were in (currentInformationState), the reward
    // received taking action (actionToTake), and the maximum qvalue in the state we ended up in
    // (newInformationState)
    qtable[currentInformationState]
      = qtable[currentInformationState]!.map {
        // Don't modify any values for other actions thank actionToTake
        if $0.action != actionToTake {
          return $0
        }

        // Get the max qvalue in the informationState we ended up in, or 0 if there are no values there
        // (e.g. for terminal states)
        var maxQValueInNewState = 0.0
        if let qvalues = qtable[newInformationState] {
          maxQValueInNewState = qvalues.max { $0.qvalue < $1.qvalue }!.qvalue
        }

        // Formula for updating the value:
        //    Keep 0.9 of original value (1 - learning rate)
        //    Mix with 0.1 (called learning rate) of the delta
        //    Only consider 0.9 (called gamma) of maxQValueInNewState
        let newQValue = (1 - 0.1) * $0.qvalue + 0.1 * (actionReward + 0.9 * maxQValueInNewState)

        let rv = ($0.action, newQValue)
        return rv
      }  // End of qvalue update
  }  // End of episode
}  // End of training over x number of episodes

mazeEnv.printMazeAndTable(header: "--- After training \(episodeCount) episodes", qtable: qtable)

// Create policy table from qtable, a dictionary of type [String: [GridMaze.Action]]
// For each informationState, this is a list of preferred actions
// The value list only has >1 elements if all those actions have equivalent (and max) qvalues
let policyTable = qtable.reduce([String: [GridMaze.Action]]()) { (dict, entry) in
  let max = entry.value.max(by: { $0.qvalue < $1.qvalue })!.qvalue
  let preferredActions = entry.value.reduce([GridMaze.Action]()) { list, entry in
    var list = list
    if entry.qvalue == max {
      list.append(entry.action)
    }
    return list
  }

  var dict = dict
  dict[entry.key] = preferredActions
  return dict
}

mazeEnv.printMazeAndTable(header: "--- Final Policy", ptable: policyTable)
