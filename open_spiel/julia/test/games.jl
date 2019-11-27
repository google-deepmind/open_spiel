@testset "Games" begin

@testset "check registered names" begin
    game_names = registered_names()

    expected = Set([
        "backgammon",
        "blotto",
        "breakthrough",
        "bridge_uncontested_bidding",
        "catch",
        "chess",
        "coin_game",
        "connect_four",
        "coop_box_pushing",
        "first_sealed_auction",
        "go",
        "goofspiel",
        "havannah",
        "hex",
        "kuhn_poker",
        "laser_tag",
        "leduc_poker",
        "liars_dice",
        "markov_soccer",
        "matching_pennies_3p",
        "matrix_cd",
        "matrix_coordination",
        "matrix_mp",
        "matrix_pd",
        "matrix_rps",
        "matrix_rpsw",
        "matrix_sh",
        "matrix_shapleys_game",
        "misere",
        "negotiation",
        "oshi_zumo",
        "oware",
        "pentago",
        "phantom_ttt",
        "pig",
        "quoridor",
        "tic_tac_toe",
        "tiny_bridge_2p",
        "tiny_bridge_4p",
        "tiny_hanabi",
        "turn_based_simultaneous_game",
        "y",
    ])

    get(ENV, "BUILD_WITH_HANABI", "OFF") == "ON" && push!(expected, "hanabi")
    get(ENV, "BUILD_WITH_ACPC", "OFF") == "ON" && push!(expected, "universal_poker")
    @test sort(game_names) == sort(collect(expected))
end

@testset "no mandatory parameters" begin
    has_mandatory_params(game_type_info) = any(is_mandatory, values(parameter_specification(game_type_info)))

    games_with_mandatory_parameters = [
        short_name(game)
        for game in registered_games()
        if has_mandatory_params(game)
    ]
    expected = [
        # Mandatory parameters prevent various sorts of automated testing.
        # Only add games here if there is no sensible default for a parameter.
        "misere",
        "turn_based_simultaneous_game",
    ]
    @test sort(games_with_mandatory_parameters) == sort(expected)
end

@testset "registered game attributes" begin
    games = Dict(short_name(game_info) => game_info for game_info in registered_games())
    @test dynamics(games["kuhn_poker"]) == SEQUENTIAL
    @test chance_mode(games["kuhn_poker"]) == EXPLICIT_STOCHASTIC
    @test information(games["kuhn_poker"]) == IMPERFECT_INFORMATION
    @test utility(games["kuhn_poker"]) == ZERO_SUM
    @test min_num_players(games["kuhn_poker"]) == 2
end

@testset "create game" begin
    game = load_game("kuhn_poker")
    game_info = get_type(game)
    @test information(game_info) == IMPERFECT_INFORMATION
    @test num_players(game) == 2
end

@testset "play kuhn_poker" begin
    game = load_game("kuhn_poker")
    state = new_initial_state(game)
    @test is_chance_node(state) == true
    @test chance_outcomes(state) == [0 => 1/3, 1 => 1/3, 2 => 1/3]

    apply_action(state, 1)
    @test is_chance_node(state) == true
    @test chance_outcomes(state) == [0 => 1/2, 2 => 1/2]

    apply_action(state, 2)
    @test is_chance_node(state) == false
    @test legal_actions(state) == [0, 1]
end

@testset "tic_tac_toe" begin
    game = load_game("tic_tac_toe")
    state = new_initial_state(game)
    @test is_chance_node(state) == false
    @test is_terminal(state) == false
    @test legal_actions(state) == 0:8
end

@testset "GameParameter" begin
    io = IOBuffer()
    print(io, GameParameter(true))
    @test String(take!(io)) == "GameParameter(bool_value=True)"
    print(io, GameParameter(false))
    @test String(take!(io)) == "GameParameter(bool_value=False)"
    print(io, GameParameter("one"))
    @test String(take!(io)) == "GameParameter(string_value='one')"
    print(io, GameParameter(1))
    @test String(take!(io)) == "GameParameter(int_value=1)"
    print(io, GameParameter(1.0))
    @test String(take!(io)) == "GameParameter(double_value=1)"
    print(io, GameParameter(1.2))
    @test String(take!(io)) == "GameParameter(double_value=1.2)"
end

end