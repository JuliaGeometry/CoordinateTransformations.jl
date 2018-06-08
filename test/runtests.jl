using Compat.Test
using Compat.LinearAlgebra
using CoordinateTransformations
using ForwardDiff: Dual, partials
using StaticArrays

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("affine.jl")
    include("perspective.jl")

end
