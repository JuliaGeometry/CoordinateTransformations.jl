using CoordinateTransformations
using BaseTestNext
using FixedSizeArrays
import ForwardDiff.GradientNumber
Base.show(io::IO, g::GradientNumber) = print(io, "GradientNumber($(g.value), $(g.partials.data))") # Output was incomprehensible!

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("commontransformations.jl")

end
