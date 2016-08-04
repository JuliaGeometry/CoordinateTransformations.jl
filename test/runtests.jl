using CoordinateTransformations
using Base.Test
using ForwardDiff: Dual, partials
using StaticArrays

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("affine.jl")

end
