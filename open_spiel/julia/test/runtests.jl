using OpenSpiel_jll
using StatsBase
using Test
using CxxWrap
using Random

function evaluate_bots(
    state::Union{Ptr{Nothing}, CxxPtr{<:State}, CxxWrap.StdLib.UniquePtrAllocated{State}},
    bots::Vector{<:Bot},
    seed::Int)

    rng = MersenneTwister(seed)

    for bot in bots
        restart_at(bot, state)
    end

    while !is_terminal(state)
        if is_chance_node(state)
            outcomes_with_probs = chance_outcomes(state)
            actions, probs = zip(outcomes_with_probs...)
            action = actions[sample(rng, weights(collect(probs)))]
            apply_action(state, action)
        elseif is_simultaneous_node(state)
            chosen_actions = [
                legal_actions(state, pid)[pid+1] ? step(bot, state) : INVALID_ACTION
                for (pid, bot) in enumerate(bots)
            ]  # in julia, index starts with 1
            apply_action(state, chosen_actions)
        else
            apply_action(state, step(bots[current_player(state) + 1], state))
        end
    end
    returns(state)
end

@testset "OpenSpiel_jll.jl" begin
    include("games_api.jl")
    include("games_simulation.jl")
    include("bots.jl")
    include("cfr.jl")
    include("trajector.jl")
end
