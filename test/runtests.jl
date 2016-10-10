using CoordinateTransformations
using Base.Test
using ForwardDiff: Dual, partials
using StaticArrays

# See https://github.com/JuliaLang/julia/issues/18858
Base.isapprox(a::UniformScaling, b::UniformScaling; kwargs...) = isapprox(a.λ, b.λ; kwargs...)

@testset "CoordinateTransformations" begin

    include("core.jl")
    include("coordinatesystems.jl")
    include("affine.jl")
    include("perspective.jl")

end
