@testset "Games" begin

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
    @test is_initial_state(state) == true
    @test chance_outcomes(state) == [0 => 1/3, 1 => 1/3, 2 => 1/3]

    apply_action(state, 1)
    @test is_chance_node(state) == true
    @test is_initial_state(state) == false
    @test chance_outcomes(state) == [0 => 1/2, 2 => 1/2]

    apply_action(state, 2)
    @test is_chance_node(state) == false
    @test is_initial_state(state) == false
    @test legal_actions(state) == [0, 1]

    @test length(full_history(state)) == 2
end

@testset "tic_tac_toe" begin
    game = load_game("tic_tac_toe")
    state = new_initial_state(game)
    @test is_chance_node(state) == false
    @test is_terminal(state) == false
    @test is_initial_state(state) == true
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

@testset "simultaneous game history" begin
    game = load_game("coop_box_pushing")
    state = new_initial_state(game)
    apply_action(state, 0)
    state2 = new_initial_state(game)
    apply_action(state2, fill(0, num_players(game)))
    @test history(state) == history(state2)
end

@testset "Matrixrame" begin
    matrix_blotto = load_matrix_game("blotto")
    @test num_rows(matrix_blotto) == 66
    @test num_cols(matrix_blotto) == 66

    kuhn_game = load_game("kuhn_poker")
    kuhn_matrix_game = extensive_to_matrix_game(kuhn_game)
    @test num_rows(kuhn_matrix_game) == 64
    @test num_cols(kuhn_matrix_game) == 64
end

end
