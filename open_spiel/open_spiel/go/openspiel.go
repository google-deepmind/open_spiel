// Package openspiel provides and API to C++ OpenSpiel.
package openspiel

// #cgo CFLAGS: -I.
// #cgo LDFLAGS: -L. -L../../build/go -lgospiel
// #include <stdlib.h>
// #include "go_open_spiel.h"
import "C" // keep
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Game is a wrapper object around an open_spiel::Game object
type Game struct {
	game unsafe.Pointer
}

// State is a wrapper arounds an open_spiel::State object
type State struct {
	state unsafe.Pointer
}

// Test prints out a nice testing message!
func Test() {
	C.Test()
}

// LoadGame loads a game!
func LoadGame(name string) *Game {
	cs := C.CString(name)
	game := &Game{C.LoadGame(cs)}
	C.free(unsafe.Pointer(cs))
	runtime.SetFinalizer(game, deleteGame)
	return game
}

// LongName returns the long name of a game
func (game *Game) LongName() string {
	cs := C.GameLongName(game.game)
	str := C.GoString(cs)
	C.free(unsafe.Pointer(cs))
	return str
}

// ShortName returns the short name of a game
func (game *Game) ShortName() string {
	cs := C.GameLongName(game.game)
	str := C.GoString(cs)
	C.free(unsafe.Pointer(cs))
	return str
}

// NewInitialState returns a new initial state.
func (game *Game) NewInitialState() *State {
	state := &State{C.GameNewInitialState(game.game)}
	runtime.SetFinalizer(state, deleteState)
	return state
}

// MaxGameLength returns the maximum length of one game.
func (game *Game) MaxGameLength() int {
	return int(C.GameMaxGameLength(game.game))
}

// NumPlayers returns the number of players in this game
func (game *Game) NumPlayers() int {
	return int(C.GameNumPlayers(game.game))
}

// NumDistinctActions returns a number of distinct actions that the game has
func (game *Game) NumDistinctActions() int {
	return int(C.GameNumDistinctActions(game.game))
}

// String returns a string representation of the state
func (state *State) String() string {
	cs := C.StateToString(state.state)
	str := C.GoString(cs)
	C.free(unsafe.Pointer(cs))
	return str
}

// IsTerminal returns whether a state is terminal or not
func (state *State) IsTerminal() bool {
	val := C.StateIsTerminal(state.state)
	return val == 1
}

// IsChanceNode returns whether a state is a chance node or not
func (state *State) IsChanceNode() bool {
	val := C.StateIsChanceNode(state.state)
	return val == 1
}

// Clone clones this state.
func (state *State) Clone() *State {
	clone := &State{C.StateClone(state.state)}
	runtime.SetFinalizer(clone, deleteState)
	return clone
}

// LegalActions returns a list of legal actions
func (state *State) LegalActions() []int {
	length := int(C.StateNumLegalActions(state.state))
	legalActions := make([]int, length)
	cppLegalActions := make([]C.int, length)
	C.StateFillLegalActions(state.state, unsafe.Pointer(&cppLegalActions[0]))
	for i := 0; i < length; i++ {
		legalActions[i] = int(cppLegalActions[i])
	}
	return legalActions
}

// LegalActionsMask returns a mask marking all legal actions as true
func (state *State) LegalActionsMask() []bool {
	length := int(C.StateNumDistinctActions(state.state))
	legalActionMask := make([]bool, length)
	cppLegalActionsMask := make([]C.int, length)
	C.StateFillLegalActionsMask(state.state, unsafe.Pointer(&cppLegalActionsMask[0]))
	for i := 0; i < length; i++ {
		legalActionMask[i] = (cppLegalActionsMask[i] > 0)
	}
	return legalActionMask
}

// Observation returns an observation as a list
func (state *State) Observation() []float32 {
	length := int(C.StateSizeObservation(state.state))
	observation := make([]float32, length)
	cppObservation := make([]C.double, length)
	C.StateFillObservation(state.state, unsafe.Pointer(&cppObservation[0]))
	for i := 0; i < length; i++ {
		observation[i] = float32(cppObservation[i])
	}
	return observation
}

// ObservationPlayer returns an observation as a list
func (state *State) ObservationPlayer(player int) []float32 {
	length := int(C.StateSizeObservation(state.state))
	observation := make([]float32, length)
	cppObservation := make([]C.double, length)
	C.StateFillObservationPlayer(state.state, unsafe.Pointer(&cppObservation[0]), C.int(player))
	for i := 0; i < length; i++ {
		observation[i] = float32(cppObservation[i])
	}
	return observation
}

// InformationState returns an observation as a list
func (state *State) InformationState() []float32 {
	length := int(C.StateSizeInformationState(state.state))
	informationState := make([]float32, length)
	cppInformationState := make([]C.double, length)
	C.StateFillInformationState(state.state, unsafe.Pointer(&cppInformationState[0]))
	for i := 0; i < length; i++ {
		informationState[i] = float32(cppInformationState[i])
	}
	return informationState
}

// InformationStateAsString returns an observation as a list
func (state *State) InformationStateAsString() string {
	infostate := state.InformationState()
	s := ""
	for _, v := range infostate {
		s = s + fmt.Sprintf("%f", v)
	}
	return s
}

// InformationStatePlayer returns an observation as a list
func (state *State) InformationStatePlayer(player int) []float32 {
	length := int(C.StateSizeInformationState(state.state))
	informationState := make([]float32, length)
	cppInformationState := make([]C.double, length)
	C.StateFillInformationStatePlayer(state.state, unsafe.Pointer(&cppInformationState[0]), C.int(player))
	for i := 0; i < length; i++ {
		informationState[i] = float32(cppInformationState[i])
	}
	return informationState
}

// InformationStatePlayerAsString returns an observation as a list
func (state *State) InformationStatePlayerAsString(player int) string {
	infostate := state.InformationStatePlayer(player)
	s := ""
	for _, v := range infostate {
		s = s + fmt.Sprintf("%f", v)
	}
	return s
}

// ChanceOutcomes returns an action slice and a probability slice
func (state *State) ChanceOutcomes() ([]int, []float32) {
	length := int(C.StateSizeChanceOutcomes(state.state))
	action := make([]int, length)
	probability := make([]float32, length)

	cppAction := make([]C.int, length)
	cppProbability := make([]C.double, length)

	C.StateFillChanceOutcomes(state.state, unsafe.Pointer(&cppAction[0]), unsafe.Pointer(&cppProbability[0]))
	for i := 0; i < length; i++ {
		action[i] = int(cppAction[i])
		probability[i] = float32(cppProbability[i])
	}
	return action, probability
}

// CurrentPlayer returns the current player to play at the state
func (state *State) CurrentPlayer() int {
	return int(C.StateCurrentPlayer(state.state))
}

// ActionToString returns a string representation of the action
func (state *State) ActionToString(player int, action int) string {
	cs := C.StateActionToString(state.state, C.int(player), C.int(action))
	str := C.GoString(cs)
	C.free(unsafe.Pointer(cs))
	return str
}

// PlayerReturn returns the return for the specified player
func (state *State) PlayerReturn(player int) float32 {
	return float32(C.StatePlayerReturn(state.state, C.int(player)))
}

// ApplyAction applies the specified action to the state
func (state *State) ApplyAction(action int) {
	C.StateApplyAction(state.state, C.int(action))
}

func deleteGame(game *Game) {
	C.DeleteGame(game.game)
}

func deleteState(state *State) {
	C.DeleteState(state.state)
}
