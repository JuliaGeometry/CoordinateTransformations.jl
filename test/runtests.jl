using Test
using LinearAlgebra
using CoordinateTransformations
using ForwardDiff: Dual, partials
using StaticArrays
using Unitful
using Documenter: doctest
using Aqua

@testset "CoordinateTransformations" begin

    doctest(CoordinateTransformations, manual=false)
    include("core.jl")
    include("coordinatesystems.jl")
    include("affine.jl")
    include("perspective.jl")
    Aqua.test_all(CoordinateTransformations)
end
