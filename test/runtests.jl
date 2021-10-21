using Test
using LinearAlgebra
using CoordinateTransformations
using ForwardDiff: Dual, partials
using StaticArrays
using Unitful
using Documenter: doctest
using BenchmarkTools


@testset "CoordinateTransformations" begin

#   doctest(CoordinateTransformations)
#   include("core.jl")
#   include("coordinatesystems.jl")
#   include("affine.jl")
#   include("perspective.jl")
    include("vectorize.jl")

end
