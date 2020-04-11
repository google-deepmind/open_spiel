module OpenSpiel

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import CxxWrap:argument_overloads

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
