using CxxWrap.StdLib: StdVector
using CxxWrap: @cxxdereference

@cxxdereference Base.show(io::IO, g::Game) = print(io, to_string(g))
@cxxdereference Base.show(io::IO, s::State) = print(io, to_string(s))
Base.show(io::IO, gp::Union{GameParameterAllocated, GameParameterDereferenced}) = print(io, to_repr_string(gp))

GameParameter(x::Int) = GameParameter(Ref(Int32(x)))

Base.copy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = deepcopy(s)
Base.deepcopy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = clone(s[])

# if Sys.KERNEL == :Linux
#     function apply_action(state, actions::AbstractVector{<:Number})
#         A = StdVector{CxxLong}()
#         for a in actions
#             push!(A, a)
#         end
#         apply_actions(state, A)
#     end
# elseif Sys.KERNEL == :Darwin
#     function apply_action(state, actions::AbstractVector{<:Number})
#         A = StdVector{Int}()
#         for a in actions
#             push!(A, a)
#         end
#         apply_actions(state, A)
#     end
# else
#     @error "unsupported system"
# end

apply_action(state, actions::AbstractVector{<:Number}) = apply_actions(state, StdVector(actions))

function deserialize_game_and_state(s::CxxWrap.StdLib.StdStringAllocated)
    game_and_state = _deserialize_game_and_state(s)
    first(game_and_state), last(game_and_state)
end

Base.values(m::StdMap) = [m[k] for k in keys(m)]

function StdMap{K, V}(kw) where {K, V}
    ps = StdMap{K, V}()
    for (k, v) in kw
        ps[convert(K, k)] = convert(V, v)
    end
    ps
end

function Base.show(io::IO, ps::StdMapAllocated{K, V}) where {K, V}
    println(io, "StdMap{$K,$V} with $(length(ps)) entries:")
    for k in keys(ps)
        println(io, "  $k => $(ps[k])")
    end
end

function load_game(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...)
    if length(kw) == 0
        _load_game(s)
    else
        ps = [StdString(string(k)) => v for (k,v) in kw]
        _load_game(s, StdMap{StdString, GameParameter}(ps))
    end
end

function load_game_as_turn_based(s::Union{String, CxxWrap.StdLib.StdStringAllocated}; kw...)
    if length(kw) == 0
        _load_game_as_turn_based(s)
    else
        ps = [StdString(string(k)) => v for (k,v) in kw]
        _load_game_as_turn_based(s, StdMap{StdString, GameParameter}(ps))
    end
end

new_initial_state(game::CxxWrap.CxxWrapCore.CxxBaseRef{Game}) = new_initial_state(game[])