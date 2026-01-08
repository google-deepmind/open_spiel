// Package main provides a simple example use of the Go API.
package main

import (
	"fmt"

	"math/rand"

	"openspiel"
)

func main() {

	openspiel.Test()

	game := openspiel.LoadGame("breakthrough")
	fmt.Printf("Game's long name is %s\n", game.LongName())

	state := game.NewInitialState()

	for !state.IsTerminal() {
		fmt.Printf("\n%s", state.String())

		curPlayer := state.CurrentPlayer()
		legalActions := state.LegalActions()
		for i := 0; i < len(legalActions); i++ {
			fmt.Printf("Legal action: %s\n", state.ActionToString(curPlayer, legalActions[i]))
		}

		sampledIdx := rand.Intn(len(legalActions))
		sampledAction := legalActions[sampledIdx]
		fmt.Printf("Sampled action: %s\n", state.ActionToString(curPlayer, sampledAction))

		state.ApplyAction(sampledAction)
	}

	fmt.Printf("\nTerminal state reached!\n")
	fmt.Printf(state.String())
	fmt.Printf("\n")
	for i := 0; i < game.NumPlayers(); i++ {
		fmt.Printf("Return for player %d is %f\n", i, state.PlayerReturn(i))
	}
}
