# Julia OpenSpiel

We also provide a Julia wrapper for the OpenSpiel project. Most APIs are aligned with those in Python (some are extended to accept `AbstractArray` and/or keyword arguments for convenience). See `spiel.h` for the full API description.

## Install

Currently this package is not registered. The reason is that we rely on some new features introduced in [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl), which is not released yet. Once CxxWrap.jl@v0.9.0 is released, we'll register this package and then you can simply install this package by `] add OpenSpiel` in the Julia REPL. For now, you need to follow the instructions bellow to install this package:

1. Install Julia and dependencies
  Edit `open_spiel/scripts/global_variables.sh` and set `BUILD_WITH_JULIA=ON`. Then run `./install.sh`. If you already have Julia installed on your system, make sure that it is visible in your terminal and its version is v1.1 or later. Otherwise, Julia v1.3 will be automatically installed in your home dir and a soft link will be created at `/usr/local/bin/julia`.

1. Build and run tests

    ```bash
    ./open_spiel/scripts/build_and_run_tests.sh
    ```

1. Install
    ```julia
    ] dev ./open_spiel/julia  # run in Julia REPL
    ```

## Example

Here we demonstrate how to use the Julia API to play one trajector:

```julia
using OpenSpiel_jll

# Here we need the StatsBase package for weighted sampling
using Pkg
Pkg.add("StatsBase")
using StatsBase

function run_once(name)
    game = load_game(name)
    state = new_initial_state(game)
    println("initial state of game[$(name)] is:\n$(state)")

    while !is_terminal(state)
        if is_chance_node(state)
            outcomes_with_probs = chance_outcomes(state)
            println("Chance node, got $(length(outcomes_with_probs)) outcomes")
            actions, probs = zip(outcomes_with_probs...)
            action = actions[sample(weights(collect(probs)))]
            println("Sampled outcome: $(action_to_string(state, action))")
            apply_action(state, action)
        elseif is_simultaneous_node(state)
            chosen_actions = [rand(legal_actions(state, pid-1)) for pid in 1:num_players(game)]  # in julia, index starts with 1
            println("Chosen actions: $([action_to_string(state, pid-1, action) for (pid, action) in enumerate(chosen_actions)])")
            apply_action(state, chosen_actions)
        else
            action = rand(legal_actions(state))
            println("Player $(current_player(state)) randomly sampled action: $(action_to_string(state, action))")
            apply_action(state, action)
        end
        println(state)
    end
    rts = returns(state)
    for pid in 1:num_players(game)
        println("Utility for player $(pid-1) is $(rts[pid])")
    end
end

run_once("tic_tac_toe")
run_once("kuhn_poker")
run_once("goofspiel(imp_info=True,num_cards=4,points_order=descending)")
```

## Q&A

1. Why is this package named `OpenSpiel_jll` but not `OpenSpiel`?

    The reason is that we plan to use [BinaryBuilder](https://github.com/JuliaPackaging/BinaryBuilder.jl) for the building process once the dependencies and APIs are stable. So by convention, this package is named `OpenSpiel_jll`. Another package named `OpenSpiel` will be registered later.

1. What is `StdVector`?

    `StdVector` is introduced in [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) recently. It is a wrapper of `std::vector` in the C++ side. Since that it is a subtype of `AbstractVector`, most functions should just work out of the box.

1. `0-based` or `1-based`?

    Considering that this package is very low level, most APIs are `0-based` to keep the code concise and consistent. Especially take care of the `Player` id which starts from `0`. And it wouldn't surprise you that some types from the Julia side, like `StdVector`, are `1-based`.

1. I can't find the `xxx` function/type in the Julia wrapper/The program exits unexpectedly.

    Although most of the functions and types should be exported, there is still a chance that some APIs are not well tested. So if you encounter any error, please do not hesitate to create an issue.