using OpenSpiel_jll
using StatsBase
using Test

@testset "OpenSpiel_jll.jl" begin
    include("games_api.jl")
    include("games_simulation.jl")
end