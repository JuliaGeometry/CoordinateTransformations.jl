using CoordinateTransformations
using BaseTestNext
using FixedSizeArrays
using ForwardDiff: Dual, partials
using Compat

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("commontransformations.jl")

end
