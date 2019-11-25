module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap

@wrapmodule(LIB_OPEN_SPIEL)

function __init__()
    @initcxx
end

end