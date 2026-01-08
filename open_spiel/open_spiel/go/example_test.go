package openspiel_test

import (
	"fmt"
	"strings"

	"openspiel"
)

func ExampleTicTacToe() {
	game := openspiel.LoadGame("tic_tac_toe")
	fmt.Println(game.LongName())
	state := game.NewInitialState()

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(4) // Middle

	stateClone := state.Clone()
	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(0) // Top-left

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(2) // Top-right

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(6) // Bottom-left

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(3) // Middle-left

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(5) // Middle-right

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(7) // Bottom

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(1) // Top

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())
	fmt.Println(state.Observation())
	state.ApplyAction(8) // Bottom-right

	fmt.Println(state.IsTerminal())
	fmt.Println(state.String())

	fmt.Printf("Player 0 return: %f\n", state.PlayerReturn(0))
	fmt.Printf("Player 1 return: %f\n", state.PlayerReturn(1))

	fmt.Println(stateClone.IsTerminal())
	fmt.Println(stateClone.String())

	// Output:
	// Tic Tac Toe
	// false
	// ...
	// ...
	// ...
	// [1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
	// false
	// ...
	// .x.
	// ...
	// [1 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
	// false
	// ...
	// .x.
	// ...
	// [1 1 1 1 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
	// false
	// o..
	// .x.
	// ...
	// [0 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
	// false
	// o.x
	// .x.
	// ...
	// [0 1 0 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0]
	// false
	// o.x
	// .x.
	// o..
	// [0 1 0 1 0 1 0 1 1 1 0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 0 0]
	// false
	// o.x
	// xx.
	// o..
	// [0 1 0 0 0 1 0 1 1 1 0 0 0 0 0 1 0 0 0 0 1 1 1 0 0 0 0]
	// false
	// o.x
	// xxo
	// o..
	// [0 1 0 0 0 0 0 1 1 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0 0 0 0]
	// false
	// o.x
	// xxo
	// ox.
	// [0 1 0 0 0 0 0 0 1 1 0 0 0 0 1 1 0 0 0 0 1 1 1 0 0 1 0]
	// false
	// oox
	// xxo
	// ox.
	// [0 0 0 0 0 0 0 0 1 1 1 0 0 0 1 1 0 0 0 0 1 1 1 0 0 1 0]
	// true
	// oox
	// xxo
	// oxx
	// Player 0 return: 0.000000
	// Player 1 return: 0.000000
	// false
	// ...
	// .x.
	// ...
}

func ExampleLoadParametrizedGame() {
	game := openspiel.LoadGame("breakthrough(rows=6,columns=6)")
	state := game.NewInitialState()
	fmt.Println(state.String())

	game = openspiel.LoadGame("turn_based_simultaneous_game(game=goofspiel(num_cards=5,imp_info=true,points_order=descending))")
	state = game.NewInitialState()
	goofStringLines := strings.Split(state.String(), "\n")
	for i := 0; i < len(goofStringLines); i++ {
		fmt.Println(strings.TrimSpace(goofStringLines[i]))
	}

	// Output:
	// 6bbbbbb
	// 5bbbbbb
	// 4......
	// 3......
	// 2wwwwww
	// 1wwwwww
	//  abcdef
	//
	// Partial joint action:
	// P0 hand: 1 2 3 4 5
	// P1 hand: 1 2 3 4 5
	// P0 actions:
	// P1 actions:
	// Point card sequence: 5
	// Points: 0 0
}
