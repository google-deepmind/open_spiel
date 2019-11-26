module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import Base:show

@wrapmodule(LIB_OPEN_SPIEL)

Base.show(io::IO, g::CxxWrap.StdLib.SharedPtrAllocated{OpenSpiel_jll.Game}) = print(io, string(g))

# export all
for n in names(@__MODULE__(); all=true)
    if Base.isidentifier(n) &&
        !startswith(String(n), "__cxxwrap") &&
        n ∉ (Symbol(@__MODULE__()), :eval, :include)
        @eval export $n
    end
end


function __init__()
    @initcxx
end

end