import TensorFlow

/// A stochastic policy assigns a probability to each action available in a given state.
public protocol StochasticPolicy {
  associatedtype Game: GameProtocol
  /// Actions and probabilities assigned by the policy for the current player in the given state.
  /// In imperfect information games, this should depend only on the player's `informationState`.
  func actionProbabilities(forState state: Game.State) -> [Game.Action: Double]
}

public protocol TensorFlowPolicy: StochasticPolicy { }

public protocol MaskedTensorFlowPolicy: TensorFlowPolicy {
  /// The logits for any illegal actions must be `-Double.infinity`.
  func maskedActionLogits(forInformationStateTensor infoState: Tensor<Double>) -> Tensor<Double>
}

extension MaskedTensorFlowPolicy {
  func actionProbabilities(
    forInformationStateTensor infoState: Tensor<Double>
  ) -> Tensor<Double> {
    return softmax(maskedActionLogits(forInformationStateTensor: infoState))
  }

  func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
    let probabilities = actionProbabilities(
      forInformationStateTensor: state.informationStateAsTensor()).scalars
    let transitions = zip(state.game.allActions, probabilities).filter { $1 != 0 }
    return Dictionary(uniqueKeysWithValues: transitions)
  }
}

public protocol UnmaskedTensorFlowPolicy: TensorFlowPolicy {
  /// The logits for illegal actions can have any values and will be ignored.
  func unmaskedActionLogits(forInformationStateTensor infoState: Tensor<Double>) -> Tensor<Double>
}

extension UnmaskedTensorFlowPolicy {
  func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
    let unmaskedLogits = unmaskedActionLogits(
      forInformationStateTensor: state.informationStateAsTensor())
    let maskedLogits = unmaskedLogits + state.legalActionsMaskAsTensor
    let probabilities = softmax(maskedLogits).scalars
    let transitions = zip(state.game.allActions, probabilities).filter { $0.1 != 0 }
    return Dictionary(uniqueKeysWithValues: transitions)
  }
}

/// A deterministic policy selects a single action for a given state.
public protocol DeterministicPolicy: StochasticPolicy {
  /// The action chosen by the policy for the current player in a given state.
  /// This should depend only on the player's information state.
  func action(forState state: Game.State) -> Game.Action
}

public extension DeterministicPolicy {
  func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
    return [action(forState: state): 1.0]
  }
}

/// A stochastic policy defined by a table of all states and actions.
/// This is a tractable representation only for games with a small state space.
public struct TabularPolicy<Game: GameProtocol>: StochasticPolicy {
  let table: [String: [Game.Action: Double]]
  public func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
    return self.table[state.informationStateString()]!
  }
}

/// A stochastic policy where the action distribution is uniform over all legal actions.
public struct UniformRandomPolicy<Game: GameProtocol>: StochasticPolicy {
  public init(_ game: Game) {}
  public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
    let legalActions = state.legalActions
    return Dictionary(uniqueKeysWithValues: legalActions.map { action in
      (action, 1.0 / Double(legalActions.count))
    })
  }
}

/// An optimizable logit table for tabular stochastic policies.
public struct TensorFlowLogitTable<Game: GameProtocol>: Differentiable {
  @noDerivative let informationStateCache: InformationStateCache<Game>
  /// Tensors of shape (number of distinct information states, number of distinct actions)
  /// for each player, containing logits whose softmax along the action dimension represents
  /// action probabilities for each of that player's possible information states.
  public var logits: [Tensor<Double>]
  public init(_ informationStateCache: InformationStateCache<Game>) {
    let game = informationStateCache.game
    let allActions = game.allActions
    logits = (0..<game.playerCount).map { _ in Tensor<Double>(zeros: [0, allActions.count]) }
    for playerID in 0..<game.playerCount {
      for informationState in informationStateCache.allInformationStates[playerID] {
        let legalActions =
          informationStateCache.legalActionsForInformationState[playerID][informationState]!
        var logitsForState = Tensor<Double>(zeros: [1, allActions.count])
        for actionIndex in allActions.indices where !legalActions[actionIndex] {
          logitsForState[0, actionIndex] -= Double.infinity
        }
        logits[playerID] = logits[playerID] ++ logitsForState
      }
    }
    self.informationStateCache = informationStateCache
  }
}

/// A tabular stochastic policy based on an optimizable logit table.
/// This is a tractable representation only for games with a small state space.
public struct TensorFlowTabularPolicy<Game: GameProtocol>: StochasticPolicy, Differentiable {
  @noDerivative let informationStateCache: InformationStateCache<Game>
  /// Tensors of shape (number of distinct information states, number of distinct actions)
  /// for each player, containing probabilities for each action in each of that player's possible
  /// information states.
  var probabilities: [Tensor<Double>] // lazy?
  @differentiable
  public init(_ logitTable: TensorFlowLogitTable<Game>) {
    informationStateCache = logitTable.informationStateCache
    probabilities = logitTable.logits.differentiableMap(softmax)
  }

  public func actionProbabilities(forState state: Game.State) -> [Game.Action : Double] {
    switch state.currentPlayer {
    case let .player(playerID):
      let informationState = state.informationStateString(for: state.currentPlayer)
      let probs = probabilities[playerID][
        informationStateCache.informationStateIndices[playerID][informationState]!]
      let transitions = zip(state.game.allActions, probs.scalars).filter { action, probability in
        probability != 0
      }
      return [Game.Action: Double](uniqueKeysWithValues: transitions)
    default:
      preconditionFailure("Policy invocation requires a real current player")
    }
  }
}

/// A stochastic policy defined by a neural network operating on information states.
public struct NeuralNetworkPolicy<Game: GameProtocol, Model: Layer>: TensorFlowPolicy, Differentiable
  where Model.Input == Tensor<Double>, Model.Output == Tensor<Double> {
  var model: Model

  /// Actions and probabilities assigned by the policy for the current player in the given state.
  /// In principle, this method could be marked `@differentiable`, but algorithms currently
  /// implemented do not yet require this.
  func unmaskedActionLogits(forInformationStateTensor infoState: Tensor<Double>) -> Tensor<Double> {
    return model(infoState)
  }

  // TODO: this shouldn't need to be duplicated
  public func actionProbabilities(forState state: Game.State) -> [Game.Action: Double] {
    let unmaskedLogits = unmaskedActionLogits(
      forInformationStateTensor: state.informationStateAsTensor())
    let maskedLogits = unmaskedLogits + state.legalActionsMaskAsTensor
    let probabilities = softmax(maskedLogits).scalars
    let transitions = zip(state.game.allActions, probabilities).filter { $0.1 != 0 }
    return Dictionary(uniqueKeysWithValues: transitions)
  }
}

public func value<Policy: StochasticPolicy>(
  for player: Player,
  in state: Policy.Game.State,
  under policy: Policy
) -> Double {
  if state.isTerminal { return state.utility(for: player) }
  let actionProbabilities = state.currentPlayer == .chance ?
    state.chanceOutcomes : policy.actionProbabilities(forState: state)
  return actionProbabilities.map { action, probability in
    return probability * value(for: player, in: state.applying(action), under: policy)
  }.reduce(0, +)
}
