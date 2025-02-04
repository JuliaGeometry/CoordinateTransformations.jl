module CoordinateTransformations

using StaticArrays
using LinearAlgebra

# Core methods
export compose, ∘, transform_deriv, transform_deriv_params, recenter
export Transformation, IdentityTransformation

# 2D coordinate systems and their transformations
export Polar
export PolarFromCartesian, CartesianFromPolar

# 3D coordinate systems and their transformations
export Spherical, Cylindrical
export SphericalFromCartesian, CartesianFromSpherical,
       CylindricalFromCartesian, CartesianFromCylindrical,
       CylindricalFromSpherical, SphericalFromCylindrical

# Common transformations
export AbstractAffineMap
export AffineMap, LinearMap, Translation
export PerspectiveMap, cameramap
export kabsch

include("core.jl")
include("coordinatesystems.jl")
include("affine.jl")
include("perspective.jl")
include("kabsch.jl")

end # module
