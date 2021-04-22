# Julia OpenSpiel

We also provide a Julia wrapper for the OpenSpiel project. Most APIs are aligned
with those in Python (some are extended to accept `AbstractArray` and/or keyword
arguments for convenience). See `spiel.h` for the full API description.

## Install

For general usage, you can install this package in the Julia REPL with
`] add OpenSpiel`. Note that this method only supports the Linux platform and
ACPC is not included. For developers, you need to follow the instructions bellow
to install this package:

1.  Install Julia and dependencies. Edit
    `open_spiel/scripts/global_variables.sh` and set
    `OPEN_SPIELOPEN_SPIEL_BUILD_WITH_JULIA=ON` (you may also turn on other
    options as you wish). Then run `./install.sh`. If you already have Julia
    installed on your system, make sure that it is visible in your terminal and
    its version is v1.3 or later. Otherwise, Julia v1.3.1 will be automatically
    installed in your home dir and a soft link will be created at
    `/usr/local/bin/julia`.

1.  Build and run tests

    ```bash
    ./open_spiel/scripts/build_and_run_tests.sh
    ```

1.  Install `] dev ./open_spiel/julia` (run in Julia REPL).

## Known Problems

1.  There's a problem when building this package on Mac with XCode v11.4 or
    above (see discussions
    [here](https://github.com/deepmind/open_spiel/pull/187#issuecomment-616540881)).
    To fix it, you need to install the latest `libcxxwrap` by following the
    instructions
    [here](https://github.com/JuliaInterop/libcxxwrap-julia#building-libcxxwrap-julia)
    after running `./install.sh`. Then make sure that the result of `julia
    --project=./open_spiel/julia -e 'using CxxWrap;
    print(CxxWrap.prefix_path())'` points to the newly built `libcxxwrap`. After
    that, build and install this package as stated above.

## Example

Here we demonstrate how to use the Julia API to play one game:

```julia
using OpenSpiel

# Here we need the StatsBase package for weighted sampling
using Pkg
Pkg.add("StatsBase")
using StatsBase

function run_once(name)
    game = load_game(name)
    state = new_initial_state(game)
    println("Initial state of game[$(name)] is:\n$(state)")

    while !is_terminal(state)
        if is_chance_node(state)
            outcomes_with_probs = chance_outcomes(state)
            println("Chance node, got $(length(outcomes_with_probs)) outcomes")
            actions, probs = zip(outcomes_with_probs...)
            action = actions[sample(weights(collect(probs)))]
            println("Sampled outcome: $(action_to_string(state, action))")
            apply_action(state, action)
        elseif is_simultaneous_node(state)
            chosen_actions = [rand(legal_actions(state, pid-1)) for pid in 1:num_players(game)]  # in Julia, indices start at 1
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

1.  What is `StdVector`?

    `StdVector` is introduced in
    [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) recently. It is a
    wrapper of `std::vector` in the C++ side. Since that it is a subtype of
    `AbstractVector`, most functions should just work out of the box.

1.  `0-based` or `1-based`?

    As this package is a low-level wrapper of OpenSpiel C++, most APIs are
    zero-based: for instance, the `Player` id starts from zero. But note that
    some bridge types, like `StdVector`, implicitly convert between indexing
    conventions, so APIs that use `StdVector` are one-based.

1.  I can't find the `xxx` function/type in the Julia wrapper/The program exits
    unexpectedly.

    Although most of the functions and types should be exported, there is still
    a chance that some APIs are not well tested. So if you encounter any error,
    please do not hesitate to create an issue.
