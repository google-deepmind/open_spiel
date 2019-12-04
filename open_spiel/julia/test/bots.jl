@testset "bots" begin

@testset "MCTSBot" begin
    UCT_C = 2.

    init_bot(game, player, max_simulations, evaluator) = MCTSBot(game, player, evaluator, UCT_C, max_simulations, 5, true, 42, false, UCT)

    @testset "can play tic_tac_toe" begin
        game = load_game("tic_tac_toe")
        max_simulations = 100
        evaluator = RandomRolloutEvaluator(20, 42)
        bot0 = init_bot(game, 0, max_simulations, CxxPtr(evaluator))
        bot1 = init_bot(game, 1, max_simulations, CxxPtr(evaluator))
        results = evaluate_bots(new_initial_state(game), [bot0, bot1], 42)
        @test results[1] + results[2] == 0
    end

    @testset "can play single player" begin
        game = load_game("catch")
        max_simulations = 100
        evaluator = RandomRolloutEvaluator(20, 42)
        bot = init_bot(game, 0, max_simulations, CxxPtr(evaluator))
        results = evaluate_bots(new_initial_state(game), [bot], 42)
        @test results[] > 0
    end

    @testset "play three player stochastic games" begin
        game = load_game("pig(players=3,winscore=20,horizon=30)")
        max_simulations = 1000
        evaluator = RandomRolloutEvaluator(20, 42)
        bot0 = init_bot(game, 0, max_simulations, CxxPtr(evaluator))
        bot1 = init_bot(game, 1, max_simulations, CxxPtr(evaluator))
        bot2 = init_bot(game, 2, max_simulations, CxxPtr(evaluator))
        results = evaluate_bots(new_initial_state(game), [bot0, bot1, bot2], 42)
        @test sum(results) == 0
    end

    function get_action_by_str(state, action_str)
        for action in legal_actions(state)
            if action_str == action_to_string(state, current_player(state), action)
                return action
            end
        end
        @error "Illegal action: $action_str"
    end

    function search_tic_tac_toe_state(initial_actions)
        game = load_game("tic_tac_toe")
        state = new_initial_state(game)
        for action_str in split(initial_actions, " ")
            apply_action(state, get_action_by_str(state, action_str))
        end
        evaluator = RandomRolloutEvaluator(20, 42)
        bot = MCTSBot(game, current_player(state), CxxPtr(evaluator), UCT_C, 10000, 10, true, 42, false, UCT)
        mcts_search(bot, state), state
    end

    @testset "solve draw" begin
        root, state = search_tic_tac_toe_state("x(1,1) o(0,0) x(2,2)")
        @test to_string(state) == "o..\n.x.\n..x"
        @test get_outcome(root)[get_player(root)+1] == 0
        for c in get_children(root)
            @test get_outcome(c)[get_player(c)+1] <= 0
        end
        best = best_child(root)[]
        @test get_outcome(best)[get_player(best)+1] == 0
    end

    @testset "solve loss" begin
        root, state = search_tic_tac_toe_state("x(1,1) o(0,0) x(2,2) o(1,0) x(2,0)")
        @test to_string(state) == "oox\n.x.\n..x"
        @test get_outcome(root)[get_player(root)+1] == -1
        for c in get_children(root)
            @test get_outcome(c)[get_player(c)+1] == -1
        end
    end

    @testset "solve win" begin
        root, state = search_tic_tac_toe_state("x(1,0) o(2,2)")
        @test to_string(state) == ".x.\n...\n..o"
        @test get_outcome(root)[get_player(root)+1] == 1
        best = best_child(root)[]
        @test get_outcome(best)[get_player(best)+1] == 1
        @test action_to_string(state, get_player(best), get_action(best)) == "x(2,0)"
    end
end

end