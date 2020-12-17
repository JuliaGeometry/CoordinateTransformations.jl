module CoordinateTransformations

using StaticArrays
using LinearAlgebra

# Core methods
export compose, ∘, transform_deriv, transform_deriv_params, recenter
export Transformation, IdentityTransformation

# 2D coordinate systems and their transformations
export Polar, Polard, AbstractPolar
export PolarFromCartesian, PolardFromCartesian, CartesianFromPolar

# 3D coordinate systems and their transformations
export Spherical, Cylindrical
export SphericalFromCartesian, CartesianFromSpherical,
       CylindricalFromCartesian, CartesianFromCylindrical,
       CylindricalFromSpherical, SphericalFromCylindrical

# Common transformations
export AbstractAffineMap
export AffineMap, LinearMap, Translation
export PerspectiveMap, cameramap

include("core.jl")
include("coordinatesystems.jl")
include("affine.jl")
include("perspective.jl")

end # module
