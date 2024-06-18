using Test
using LinearAlgebra
using CoordinateTransformations
using ForwardDiff: Dual, partials
using StaticArrays
using Unitful
using Documenter
using Aqua

@testset "CoordinateTransformations" begin
    include("core.jl")
    include("coordinatesystems.jl")
    include("affine.jl")
    include("perspective.jl")

    Aqua.test_all(CoordinateTransformations)

    DocMeta.setdocmeta!(
        CoordinateTransformations,
        :DocTestSetup,
        :(using CoordinateTransformations),
        recursive=true,
    )
    doctest(CoordinateTransformations, manual=true)
end
