Base.convert(::Type{CxxWrap.StdLib.SharedPtrAllocated{Evaluator}}, p::CxxWrap.StdLib.SharedPtrAllocated{Evaluator}) = p

Base.print(io::IO, s::CxxWrap.StdLib.StdStringAllocated) = write(io, [reinterpret(UInt8, s[i]) for i in 1:length(s)])

Base.show(io::IO, g::CxxWrap.StdLib.SharedPtrAllocated{Game}) = print(io, to_string(g))
Base.show(io::IO, s::CxxWrap.StdLib.UniquePtrAllocated{State}) = print(io, to_string(s))
Base.show(io::IO, gp::Union{GameParameterAllocated, GameParameterDereferenced}) = print(io, to_repr_string(gp))

# a workaround to disable argument_overloads for bool
GameParameter(x::Bool) = GameParameter(UInt8[x])
GameParameter(x::Int) = GameParameter(Ref(Int32(x)))

Base.copy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = deepcopy(s)
Base.deepcopy(s::CxxWrap.StdLib.UniquePtrAllocated{State}) = clone(s)

if Sys.KERNEL == :Linux
    function apply_action(state, actions::AbstractVector{<:Number})
        A = StdVector{CxxLong}()
        for a in actions
            push!(A, a)
        end
        apply_actions(state, A)
    end
elseif Sys.KERNEL == :Darwin
    function apply_action(state, actions::AbstractVector{<:Number})
        A = StdVector{Int}()
        for a in actions
            push!(A, a)
        end
        apply_actions(state, A)
    end
else
    @error "unsupported system"
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
