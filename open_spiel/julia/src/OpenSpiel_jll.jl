module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import CxxWrap:argument_overloads
import Base:show, length, getindex, setindex!, keys, values, copy, deepcopy, first, last

@readmodule(LIB_OPEN_SPIEL)
@wraptypes
CxxWrap.argument_overloads(t::Type{Float64}) = Type[]
@wrapfunctions

Base.show(io::IO, g::CxxWrap.StdLib.SharedPtrAllocated{Game}) = print(io, to_string(g))
Base.show(io::IO, s::CxxWrap.StdLib.UniquePtrAllocated{State}) = print(io, to_string(s))
Base.show(io::IO, gp::Union{GameParameterAllocated, GameParameterDereferenced}) = print(io, to_repr_string(gp))

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

function deserialize_game_and_state(s::CxxWrap.StdLib.StdStringAllocated)
    game_and_state = _deserialize_game_and_state(s)
    first(game_and_state), last(game_and_state)
end

function StdMap{K, V}(kw) where {K, V}
    ps = StdMap{K, V}()
    for (k, v) in kw
        ps[convert(K, k)] = convert(V, v)
    end
    ps
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ps::StdMapAllocated{K, V}) where {K, V}
    ps_pairs = ["$k => $v" for (k, v) in zip(keys(ps), values(ps))]
    println(io, "StdMap{$K,$V} with $(length(ps_pairs)) entries:")
    for s in ps_pairs
        println(io, "  $s")
    end
end

load_game(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...) = length(kw) == 0 ? _load_game(s) : _load_game(s, StdMap{StdString, GameParameter}([StdString(string(k)) => v for (k,v) in kw]))

load_game_as_turn_based(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...) = length(kw) == 0 ? _load_game_as_turn_based(s) : _load_game_as_turn_based(s, StdMap{StdString, GameParameter}([StdString(string(k)) => v for (k,v) in kw]))

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