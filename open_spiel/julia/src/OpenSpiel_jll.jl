module OpenSpiel_jll

include("$(@__DIR__)/../deps/deps.jl")

using CxxWrap
import CxxWrap:argument_overloads
import Base:show, length, getindex, setindex!, keys, values, copy, deepcopy, first, last

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

function deserialize_game_and_state(s::CxxWrap.StdLib.StdStringAllocated)
    game_and_state = _deserialize_game_and_state(s)
    first(game_and_state), last(game_and_state)
end

function GameParameters(kw::Iterators.Pairs)
    ps = GameParameters()
    for (k, v) in kw
        ps[string(k)] = v
    end
    ps
end

function Base.show(io::IO, ps::GameParametersAllocated)
    ps_pairs = ["$k => $v" for (k, v) in zip(keys(ps), values(ps))]
    s = length(ps_pairs) == 0 ? "" : join(',', ps_pairs)
    print(io, "GameParameters($s)")
end

load_game(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...) = length(kw) == 0 ? _load_game(s) : _load_game(s, GameParameters(kw))

load_game_as_turn_based(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...) = length(kw) == 0 ? _load_game_as_turn_based(s) : _load_game_as_turn_based(s, ps)

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