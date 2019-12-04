module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import CxxWrap:argument_overloads
import Base:show, length, getindex, setindex!, keys, values, copy, deepcopy, first, last, step, getfield, setfield!


@wrapmodule(LIB_OPEN_SPIEL)

include("patch.jl")

# export all
for n in names(@__MODULE__(); all=true)
    if Base.isidentifier(n) &&
        !startswith(String(n), "_") &&
        n ∉ (Symbol(@__MODULE__()), :eval, :include)
        @eval export $n
    end
end


function __init__()
    @initcxx
end

end