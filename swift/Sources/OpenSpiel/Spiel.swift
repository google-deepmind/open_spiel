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

import TensorFlow

/// Represents a parameterized instance of a game, for example Breakthrough(8x8),
/// but not the state of gameplay (instances are therefore immutable).
public protocol GameProtocol {
  associatedtype State: StateProtocol where State.Game == Self
  associatedtype Action: Hashable

  /// The initial state for the game.
  var initialState: State { get }

  /// Static information on the game type.
  static var info: GameInfo { get }

  /// The number of players in this instantiation of the game.
  /// Does not include the chance-player.
  var playerCount: Int { get }

  /// All distinct actions possible in the game for any non-chance player. This
  /// is not the same as the legal actions in any particular state as distinct
  /// actions are independent of the context (state), and often independent of
  /// the player as well. So, for instance in Tic-Tac-Toe there are 9 of these, one
  /// for each square. In games where pieces move, like e.g. Breakthrough, then
  /// there would be 64*6*2, since from an 8x8 board a single piece could only ever
  /// move to at most 6 places, and it can be a regular move or a capture move.
  /// Note: chance node outcomes are not included in this count.
  /// For example, this corresponds to the actions represented by each output
  /// neuron of the policy net head learning which move to play.
  var allActions: [Action] { get }

  /// Utility range. These functions define the lower and upper bounds on the
  /// values returned by `State.return`. This range should be as tight as possible;
  /// the intention is to give some information to algorithms that require it,
  /// and so their performance may suffer if the range is not tight. Loss/win/draw
  /// outcomes are common among games and should use the standard values of {-1,0,1}.
  var minUtility: Double { get }
  var maxUtility: Double { get }
  /// The total utility for all players. Should return `0.0` if the game is zero-sum.
  ///
  /// Note: not all games are zero-sum (e.g. cooperative games), as a result, this is an
  /// optional value.
  var utilitySum: Double? { get }

  /// Describes the structure of the information state representation in a
  /// tensor-like format. This is especially useful for experiments involving
  /// reinforcement learning and neural networks.
  /// Note: the actual information is returned in a 1-D vector by
  /// `ImperfectInformationState.informationStateAsNormalizedVector` -
  /// see the documentation of that function for details of the data layout.
  var informationStateNormalizedVectorShape: [Int] { get }

  /// Maximum length of any one game (in terms of number of decision nodes
  /// visited in the game tree). For a simultaneous action game, this is the
  /// maximum number of joint decisions. In a turn-based game, this is the
  /// maximum number of individual decisions summed over all players. Outcomes
  /// of chance nodes are not included in this length.
  var maxGameLength: Int { get }
}

/// Different kinds of "player."
public enum Player: Hashable {
  /// An actual player in the game.
  case player(Int)
  /// The "player" representing chance decisions.
  case chance
  /// All players acting at the same time.
  case simultaneous
  /// Invalid or no player.
  case invalid
  /// The current player after the game is over.
  case terminal
}

/// Represents a state in a game. Each game type has an associated state type.
public protocol StateProtocol: Hashable {
  associatedtype Game: GameProtocol where Game.State == Self

  /// The corresponding game instance.
  var game: Game { get }

  /// The current player.
  var currentPlayer: Player { get }

  /// Is this a terminal state? (i.e. has the game ended?)
  var isTerminal: Bool { get }

  /// Change the state of the game by applying the specified action in turn-based
  /// games. This function encodes the logic of the game rules. Returns true
  /// on success. In simultaneous games, returns false (`applyActions` should be
  /// used in that case.)
  ///
  /// In the case of chance nodes, the behavior of this function depends on
  /// `GameInfo.chanceMode`. If `.explicitStochastic`, then the outcome should be
  /// directly applied. If `.sampledStochastic`, then a dummy outcome is passed and
  /// the sampling of an outcome should be done in this function and then applied.
  mutating func apply(_ action: Game.Action)

  /// An array of the same length as `game.allActions` representing which of those
  /// actions are legal for the current player in this state. Not valid in chance nodes.
  var legalActionsMask: [Bool] { get }

  /// Get the chance outcomes at this state and their probabilities.
  ///
  /// Note: what is returned here depending on the game's chanceMode (in
  /// its GameInfo):
  ///   - Option 1. `.explicitStochastic`. All chance node outcomes are returned along
  ///     with their respective probabilities. Then State.apply(...) is deterministic.
  ///   - Option 2. `.sampledStochastic`. Return a dummy single action here with
  ///     probability 1, and then `State.apply(_:)` does the real sampling. In this
  ///     case, the game has to maintain its own RNG.
  var chanceOutcomes: [Game.Action: Double] { get }

  /// The list of actions leading to the state.
  var history: [Game.Action] { get }

  /// The total reward ("utility" or "return") for `player` in the current state.
  /// For games that only have a final reward, it should be 0 for all
  /// non-terminal states and the terminal utility for the final state.
  func utility(for player: Player) -> Double

  /// For imperfect information games. Returns an identifier for the current
  /// information state for the specified player.
  /// Different ground states can yield the same information state for a player
  /// when the only part of the state that differs is not observable by that
  /// player (e.g. opponents' cards in Poker.)
  ///
  /// Games that do not have imperfect information do not need to implement
  /// these methods, but most algorithms intended for imperfect information
  /// games will work on perfect information games provided the informationState
  /// is returned in a form they support. For example, informationState
  /// could simply return history for a perfect information game.
  ///
  /// The `informationState` must be returned at terminal states, since this is
  /// required in some applications (e.g. final observation in an RL
  /// environment).
  ///
  /// Not valid in chance states.
  func informationState(for player: Player) -> String

  /// Vector form, useful for neural-net function approximation approaches.
  /// The size of the vector must match Game.informationStateNormalizedVectorShape
  /// with values in lexicographic order. E.g. for 2x4x3, order would be:
  /// (0,0,0), (0,0,1), (0,0,2), (0,1,0), ... , (1,3,2).
  ///
  /// There are currently no use-case for calling this function with
  /// `Player.chance`. Thus, games are expected to raise an error in that case.
  func informationStateAsNormalizedVector(for player: Player) -> [Double]
}

public extension StateProtocol {
  /// Return an empty dictionary by default (i.e. for games without chance nodes).
  var chanceOutcomes: [Game.Action: Double] { [:] }

  /// The state resulting from applying the specified action.
  func applying(_ action: Game.Action) -> Self {
    var child = self
    child.apply(action)
    return child
  }

  /// The information state for the current player.
  func informationState() -> String {
    return informationState(for: currentPlayer)
  }

  /// The normalized-vector information state for the current player as a Swift array.
  func informationStateAsNormalizedVector() -> [Double] {
    return informationStateAsNormalizedVector(for: currentPlayer)
  }

  /// The normalized-vector information state as a TensorFlow tensor.
  /// Includes a singleton batch dimension :(
  func informationStateAsTensor(for player: Player) -> Tensor<Double> {
    return Tensor(
      shape: TensorShape([1] + game.informationStateNormalizedVectorShape),
      scalars: informationStateAsNormalizedVector(for: player))
  }

  /// The normalized-vector information state for the current player as a TensorFlow tensor.
  func informationStateAsTensor() -> Tensor<Double> {
    return informationStateAsTensor(for: currentPlayer)
  }

  /// All actions that are legal for the current player in this state. The default
  /// implementation assumes no chance actions.
  var legalActions: [Game.Action] {
    switch currentPlayer {
    case .chance:
      return Array(chanceOutcomes.keys)
    default:
      return zip(game.allActions, legalActionsMask).filter { $0.1 }.map { $0.0 }
    }
  }

  /// A tensor containing zeros for legal actions and `-Double.infinity` for illegal actions.
  /// Useful when masking logits.
  var legalActionsMaskAsTensor: Tensor<Double> {
    return Tensor(legalActionsMask.map { $0 ? 0 : -Double.infinity })
  }
  
  /// Given a game instance, `State`s can be identified by their history.
  /// Some games (e.g., with cycles) may choose to override this.
  func hash(into hasher: inout Hasher) {
    hasher.combine(history)
  }
  
  static func == (lhs: Self, rhs: Self) -> Bool {
    return lhs.history == rhs.history
  }
}

/// Static information for a game. This will determine what algorithms are
/// applicable. For example, minimax search is only applicable to two-player,
/// zero-sum games with perfect information. (Though can be made applicable to
/// games that are constant-sum.)
///
/// The number of players is not considered part of this static game type,
/// because this depends on the parameterization. See `Game.playerCount`.
public struct GameInfo {
  /// A short name with no spaces that uniquely identifies the game, e.g.
  /// "msoccer". This is the key used to distinguish games.
  public let shortName: String
  /// A long human-readable name, e.g. "Markov Soccer".
  public let longName: String

  /// Is the game one-player-at-a-time or do players act simultaneously?
  public enum Dynamics {
    /// Every player acts at each stage.
    case simultaneous
    /// Turn-based games.
    case sequential
  }
  public let dynamics: Dynamics

  /// Are there any chance nodes? If so, how is chance treated?
  /// Either all possible chance outcomes are explicitly returned as
  /// `chanceOutcomes`, and the result of `apply(_:)` is deterministic. Or
  /// just one chance outcome is returned, and the result of `apply(_:)` is
  /// stochastic.
  public enum ChanceMode {
    /// No chance nodes
    case deterministic
    /// Has at least one chance node, all with deterministic `apply(_:)`
    case explicitStochastic
    /// At least one chance node with non-deterministic `apply(_:)`
    case sampledStochastic
  }
  public let chanceMode: ChanceMode

  /// The information type of the game.
  public enum Information {
    /// aka Normal-form games (single simultaneous turn).
    case oneShot
    /// All players know the state of the game.
    case perfect
    /// Some information is hidden from some players.
    case imperfect
  }
  public let information: Information

  /// Whether the game has any constraints on the player utilities.
  public enum Utility {
    /// Utilities of all players sum to 0
    case zeroSum
    /// Utilities of all players sum to a constant
    case constantSum
    /// Total utility of all players differs in different outcomes
    case generalSum
    /// Every player gets an identical value (cooperative game).
    case identical
  }
  public let utility: Utility

  /// When are rewards handed out? Note that even if the game only specifies
  /// utilities at terminal states, the default implementation of `State.rewards`
  /// should work for RL uses (giving 0 everywhere except terminal states).
  public enum RewardModel {
    /// RL-style func r(s, a, s') via `State.rewards()` call at s'.
    case rewards
    /// Games-style, only at terminals. Call `State.returns()`.
    case terminal
  }
  public let rewardModel: RewardModel

  /// How many players can play the game. If the number can vary, the actual
  /// instantiation of the game should specify how many players there are.
  public let maxPlayers, minPlayers: Int

  /// Which type of information state representations are supported?
  public let providesInformationState: Bool
  public let providesInformationStateAsNormalizedVector: Bool
}

/// Used to sample a policy. Can also sample from chance outcomes.
/// Probabilities of the actions must sum to 1.
/// The parameter z should be a sample from a uniform distribution on the range
/// [0, 1).
public func sampleChanceOutcome<Action: Hashable>(
  _ outcomes: [Action: Double],
  z: Double = Double.random(in: 0..<1)
) -> Action {
  // First do a check that this is indeed a proper discrete distribution.
  precondition(outcomes.values.allSatisfy { 0...1 ~= $0 })
  precondition(outcomes.values.reduce(0, +) == 1.0)

  // Now sample an outcome.
  var sum = 0.0
  for (action, probability) in outcomes {
    if case sum..<sum + probability = z {
      return action
    }
    sum += probability
  }

  // If we get here, something has gone wrong
  fatalError("Internal error: failed to sample an outcome")
}

/// Iterate over all states accessible from `state` and apply function `body` to the state.
public func forEachState<State: StateProtocol>(
  from root: State,
  includeTerminals: Bool = false,
  includeChanceStates: Bool = false,
  _ body: (State) -> ()
) {
  var workList = [root]
  var seen = Set<State>()
  while let state = workList.popLast() {
    if seen.contains(state) { continue }
    if !includeTerminals && state.isTerminal { continue }
    if state.currentPlayer != .chance || includeChanceStates {
      body(state)
    }
    seen.insert(state)
    for action in state.legalActions {
      workList.append(state.applying(action))
    }
  }
}
