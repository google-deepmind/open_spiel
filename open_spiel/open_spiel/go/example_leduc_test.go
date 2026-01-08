package openspiel_test

import (
	"fmt"
	"strings"

	"openspiel"
)

func toString(s *openspiel.State) string {
	return strings.ReplaceAll(s.String(), " \n", "\n")
}

func ExampleLeduc() {
	game := openspiel.LoadGame("leduc_poker")
	fmt.Println(game.LongName())
	state := game.NewInitialState()

	// Chance node
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	action0, probability0 := state.ChanceOutcomes()
	fmt.Println(action0)
	fmt.Println(probability0)
	fmt.Print(toString(state))
	fmt.Println(state.LegalActions())
	state.ApplyAction(4)

	// Chance node
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	action1, probability1 := state.ChanceOutcomes()
	fmt.Println(action1)
	fmt.Println(probability1)
	fmt.Print(toString(state))
	fmt.Println(state.LegalActions())
	state.ApplyAction(3)

	stateClone := state.Clone()

	// player 0
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))
	fmt.Println(state.Observation())
	fmt.Println(state.ObservationPlayer(0))
	fmt.Println(state.ObservationPlayer(1))
	fmt.Println(state.InformationState())
	fmt.Println(state.InformationStatePlayer(0))
	fmt.Println(state.InformationStatePlayer(1))
	fmt.Println(state.LegalActions())
	state.ApplyAction(1)

	// player 1
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))
	fmt.Println(state.Observation())
	fmt.Println(state.ObservationPlayer(0))
	fmt.Println(state.ObservationPlayer(1))
	fmt.Println(state.InformationState())
	fmt.Println(state.InformationStatePlayer(0))
	fmt.Println(state.InformationStatePlayer(1))
	fmt.Println(state.LegalActions())
	state.ApplyAction(1)

	// Chance node
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	action2, probability2 := state.ChanceOutcomes()
	fmt.Println(action2)
	fmt.Println(probability2)
	fmt.Print(toString(state))
	fmt.Println(state.LegalActions())
	state.ApplyAction(1)

	// player 0
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))
	fmt.Println(state.Observation())
	fmt.Println(state.ObservationPlayer(0))
	fmt.Println(state.ObservationPlayer(1))
	fmt.Println(state.InformationState())
	fmt.Println(state.InformationStatePlayer(0))
	fmt.Println(state.InformationStatePlayer(1))
	fmt.Println(state.LegalActions())
	state.ApplyAction(1)

	// player 1
	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))
	fmt.Println(state.Observation())
	fmt.Println(state.ObservationPlayer(0))
	fmt.Println(state.ObservationPlayer(1))
	fmt.Println(state.InformationState())
	fmt.Println(state.InformationStatePlayer(0))
	fmt.Println(state.InformationStatePlayer(1))
	fmt.Println(state.LegalActions())
	state.ApplyAction(1)

	fmt.Println(state.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))

	fmt.Printf("Player 0 return: %f\n", state.PlayerReturn(0))
	fmt.Printf("Player 1 return: %f\n", state.PlayerReturn(1))

	fmt.Println(stateClone.IsTerminal())
	fmt.Println(state.IsChanceNode())
	fmt.Print(toString(state))

	// Output:
	// Leduc Poker
	// false
	// true
	// [0 1 2 3 4 5]
	// [0.16666667 0.16666667 0.16666667 0.16666667 0.16666667 0.16666667]
	// Round: 1
	// Player: -1
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): -10000 -10000 -10000
	// Round 1 sequence:
	// Round 2 sequence:
	// [0 1 2 3 4 5]
	// false
	// true
	// [0 1 2 3 5]
	// [0.2 0.2 0.2 0.2 0.2]
	// Round: 1
	// Player: -1
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): -10000 4 -10000
	// Round 1 sequence:
	// Round 2 sequence:
	// [0 1 2 3 5]
	// false
	// false
	// Round: 1
	// Player: 0
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): -10000 4 3
	// Round 1 sequence:
	// Round 2 sequence:
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 2]
	// false
	// false
	// Round: 1
	// Player: 1
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): -10000 4 3
	// Round 1 sequence: Call
	// Round 2 sequence:
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 2]
	// false
	// true
	// [0 1 2 5]
	// [0.25 0.25 0.25 0.25]
	// Round: 2
	// Player: -1
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): -10000 4 3
	// Round 1 sequence: Call, Call
	// Round 2 sequence:
	// [0 1 2 5]
	// false
	// false
	// Round: 2
	// Player: 0
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): 1 4 3
	// Round 1 sequence: Call, Call
	// Round 2 sequence:
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// [1 2]
	// false
	// false
	// Round: 2
	// Player: 1
	// Pot: 2
	// Money (p1 p2 ...): 99 99
	// Cards (public p1 p2 ...): 1 4 3
	// Round 1 sequence: Call, Call
	// Round 2 sequence: Call
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 1]
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 1]
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0]
	// [1 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0]
	// [0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0]
	// [1 2]
	// true
	// false
	// Round: 2
	// Player: 1
	// Pot: 0
	// Money (p1 p2 ...): 101 99
	// Cards (public p1 p2 ...): 1 4 3
	// Round 1 sequence: Call, Call
	// Round 2 sequence: Call, Call
	// Player 0 return: 1.000000
	// Player 1 return: -1.000000
	// false
	// false
	// Round: 2
	// Player: 1
	// Pot: 0
	// Money (p1 p2 ...): 101 99
	// Cards (public p1 p2 ...): 1 4 3
	// Round 1 sequence: Call, Call
	// Round 2 sequence: Call, Call
}
