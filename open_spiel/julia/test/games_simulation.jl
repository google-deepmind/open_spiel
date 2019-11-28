@testset "games simulation" begin

MAX_ACTIONS_PER_GAME = 1000

SPIEL_GAMES_LIST = registered_games()

_has_mandatory_params(x) = any(is_mandatory, values(parameter_specification(x)))

SPIEL_LOADABLE_GAMES_LIST = [
    g for g in SPIEL_GAMES_LIST if !_has_mandatory_params(g)
]

@test length(SPIEL_LOADABLE_GAMES_LIST) >= 38

SPIEL_SIMULTANEOUS_GAMES_LIST = [
    g for g in SPIEL_LOADABLE_GAMES_LIST
    if dynamics(g) == SIMULTANEOUS
]

@test length(SPIEL_SIMULTANEOUS_GAMES_LIST) >= 14

SPIEL_MULTIPLAYER_GAMES_LIST = [
    (g, p)
    for g in SPIEL_LOADABLE_GAMES_LIST
    for p in max(min_num_players(g), 2) : min(max_num_players(g), 6)
    if (max_num_players(g) > 2) &&
        (max_num_players(g) > min_num_players(g)) &&
        (short_name(g) != "tiny_hanabi")  # default payoff only works for 2p
]

@test length(SPIEL_MULTIPLAYER_GAMES_LIST) >= 35

function apply_action_test_clone(state, action)
    state_copy = copy(state)
    @test string(state) == string(state_copy)
    @test history(state) == history(state_copy)

    apply_action(state, action)
    apply_action(state_copy, action)

    @test string(state) == string(state_copy)
    @test history(state) == history(state_copy)
end

function serialize_deserialize(game, state)
    ser_str = serialize_game_and_state(game, state)
    new_game, new_state = deserialize_game_and_state(ser_str)
    @test string(game) == string(new_game)
    @test string(state) == string(new_state)
end

function simulate_game(game)
    @info "simulating game $(short_name(get_type(game)))"
    min_u, max_u = min_utility(game), max_utility(game)
    @test min_u < max_u

    state = new_initial_state(game)
    total_actions = 0

    next_serialize_check = 1

    while !is_terminal(state) && (total_actions <= MAX_ACTIONS_PER_GAME)
        total_actions += 1

        # Serialize/Deserialize is costly. Only do it every power of 2 actions.
        if total_actions >= next_serialize_check
            serialize_deserialize(game, state)
            next_serialize_check *= 2
        end

        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if is_chance_node(state)
            # Chance node: sample an outcome
            outcomes = chance_outcomes(state)
            @test length(outcomes) > 0
            action_list, prob_list = zip(outcomes...)
            action = action_list[sample(weights(collect(prob_list)))]
            apply_action(state, action)
        elseif is_simultaneous_node(state)
            chosen_actions = [
                rand(legal_actions(state, pid-1))
                for pid in 1:num_players(game)
            ]  # in julia, index starts with 1
            # Apply the joint action and test cloning states.
            apply_action_test_clone(state, chosen_actions)
        else
            # Decision node: sample action for the single current player
            action = rand(legal_actions(state, current_player(state)))
            # Apply action and test state cloning.
            apply_action_test_clone(state, action)
        end
    end

    @test total_actions > 0

    if is_terminal(state)
        # Check there are no legal actions.
        @test length(legal_actions(state)) == 0
        for player in 0:(num_players(game)-1)
            @test length(legal_actions(state, player)) == 0
        end

        utilities = returns(state)

        for u in utilities
            @test u >= min_utility(game)
            @test u <= max_utility(game)
        end

        @info "Simulation of game $game" total_actions utilities
    else
        @info "Simulation of game $game terminated after maximum number of actions $MAX_ACTIONS_PER_GAME"
    end
end

for game_info in SPIEL_LOADABLE_GAMES_LIST
    game = load_game(short_name(game_info))
    @test num_players(game) >= min_num_players(game_info)
    @test num_players(game) <= max_num_players(game_info)
    simulate_game(game)
end

for game_info in SPIEL_SIMULTANEOUS_GAMES_LIST
    converted_game = load_game_as_turn_based(short_name(game_info))
    simulate_game(converted_game)
end

for (game_info, n) in SPIEL_MULTIPLAYER_GAMES_LIST
    game = load_game(short_name(game_info); players=GameParameter(n))
    simulate_game(game)
end

simulate_game(load_game("breakthrough(rows=6,columns=6)"))
simulate_game(load_game("pig(players=2,winscore=15)"))

end