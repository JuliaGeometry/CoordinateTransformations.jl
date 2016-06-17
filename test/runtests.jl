using CoordinateTransformations
using BaseTestNext
using FixedSizeArrays
using ForwardDiff: Dual, partials

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("commontransformations.jl")

end
