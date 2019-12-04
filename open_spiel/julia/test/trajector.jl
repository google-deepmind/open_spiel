@testset "Trajectory" begin

function get_states_to_indices(game)
    state_index = StdMap{StdString, Cint}()
    to_visit = []
    push!(to_visit, new_initial_state(game))
    i = 0
    while length(to_visit) != 0
        state = pop!(to_visit)
        if (!is_chance_node(state)) && (!is_terminal(state))
            state_index[information_state_string(state)] = Cint(i)
        end
        i += 1
        for action in legal_actions(state)
            push!(to_visit, child(state, action))
        end
    end
    state_index
end

@testset "BatchedTrajectory" begin
    for game_name in ["kuhn_poker", "leduc_poker", "liars_dice"]
        game = load_game(game_name)
        batch_size = 32
        states_to_inds = get_states_to_indices(game)
        policies = StdVector([get_uniform_policy(game) for _ in 1:2])
        t = record_batched_trajectories(game, policies, states_to_inds, batch_size, false, 123, -1)
        @test length(legal_actions(t)) == batch_size
        @test length(actions(t)) == batch_size
        @test length(player_policies(t)) == batch_size
        @test length(player_ids(t)) == batch_size
        @test length(next_is_terminal(t)) == batch_size
    end
end

end