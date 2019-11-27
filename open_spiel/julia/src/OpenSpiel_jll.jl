module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import CxxWrap:argument_overloads
import Base:show, length, getindex, setindex!, keys, values, copy, deepcopy

@readmodule(LIB_OPEN_SPIEL)
@wraptypes
CxxWrap.argument_overloads(t::Type{Float64}) = Type[]
@wrapfunctions

Base.show(io::IO, g::CxxWrap.StdLib.SharedPtrAllocated{OpenSpiel_jll.Game}) = print(io, to_string(g))
Base.show(io::IO, gp::GameParameterAllocated) = print(io, to_repr_string(gp))

# a workaround to disable argument_overloads for bool
GameParameter(x::Bool) = GameParameter(UInt8[x])
GameParameter(x::Int) = GameParameter(Ref(Int32(x)))

Base.copy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = deepcopy(s)
Base.deepcopy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = clone(s)

function apply_action(state, actions::AbstractVector{<:Number})
    A = StdVector{CxxLong}()
    for a in actions
        push!(A, a)
    end
    apply_actions(state, A)
end

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