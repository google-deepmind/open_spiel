@testset "cfr" begin

function test_nash_kuhn_poker(game, policy)
    game_values = expected_returns(new_initial_state(game), policy, -1)

    # 1/18 is the Nash value. See https://en.wikipedia.org/wiki/Kuhn_poker
    nash_value = 1.0 / 18.0
    eps = 1e-3
    @test length(game_values) == 2
    @test isapprox(game_values[1], -nash_value, atol=eps)
    @test isapprox(game_values[2], nash_value, atol=eps)
end

test_exploitability_kuhn_poker(game, policy) = @test exploitability(game, policy) <= 0.05

@testset "CFRSolver" begin
    game = load_game("kuhn_poker")
    solver = CFRSolver(game)
    for _ in 1:300
        evaluate_and_update_policy(solver)
    end
    avg_policy = average_policy(solver)
    test_nash_kuhn_poker(game, avg_policy)
    test_exploitability_kuhn_poker(game, avg_policy)
end

@testset "CFRPlusSolver" begin
    game = load_game("kuhn_poker")
    solver = CFRPlusSolver(game)
    for _ in 1:200
        evaluate_and_update_policy(solver)
    end
    avg_policy = average_policy(solver)
    test_nash_kuhn_poker(game, avg_policy)
    test_exploitability_kuhn_poker(game, avg_policy)
end

end
