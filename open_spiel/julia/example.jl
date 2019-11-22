include("./OpenSpiel.jl")

using StatsBase

if length(ARGS) == 0
    game_name = "tic_tac_toe"
elseif length(ARGS) == 1
    game_name = ARGS[1]
else
    @error "unknown argument list, example usage is `julia open_spiel/julia/example.jl tic_tac_toe`"
end

println("loading game $game_name ...")
game = Spiel.load_game(game_name)
state = Spiel.new_initial_state(game)
println("initial state is\n $(Spiel.string(state))")

while !Spiel.is_terminal(state)
    if Spiel.is_chance_node(state)
        outcomes_with_probs = Spiel.chance_outcomes(state)
        println("Chance node, got $(length(outcomes_with_probs)) outcomes")
        actions, probs = zip(outcomes_with_probs...)
        action = actions[sample(weights(collect(probs)))]
        println("Sampled outcome: $(Spiel.action_to_string(state, action))")
        Spiel.apply_action(state, action)
    elseif Spiel.is_simultaneous_node(state)
        chosen_actions = [rand(Spiel.legal_actions(state, pid-1)) for pid in 1:Spiel.num_players(game)]  # in julia, index starts with 1
        println("Chosen actions: $([Spiel.action_to_string(state, pid-1, action) for (pid, action) in enumerate(chosen_actions)])")
        Spiel.apply_actions(state, chosen_actions)
    else
        action = rand(Spiel.legal_actions(state))
        println("Player $(Spiel.current_player(state)) randomly sampled action: $(Spiel.action_to_string(state, action))")
        Spiel.apply_action(state, action)
    end
    println(Spiel.string(state))
end

returns = Spiel.returns(state)
for pid in 1:Spiel.num_players(game)
    println("Utility for player $(pid-1) is $(returns[pid])")
end