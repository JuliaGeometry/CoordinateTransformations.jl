module CoordinateTransformations

using FixedSizeArrays
export Point # Use Point{N, T} from FixedSizedArrays for Cartesian frames

# Core methods
export transform, compose, ∘, transform_deriv, transform_deriv_params
export AbstractTransformation, IdentityTransformation

# 2D coordinate systems and their transformations
export Polar
export PolarFromCartesian, CartesianFromPolar

# 3D coordinate systems and their transformations
export Spherical, Cylindrical
export SphericalFromCartesian, CartesianFromSpherical,
       CylindricalFromCartesian, CartesianFromCylindrical,
       CylindricalFromSpherical, SphericalFromCylindrical

# Common transformations
export Translation
export RotationPolar, Rotation2D

#export RigidBodyTransformation, AffineTransformation

include("core.jl")
include("coordinatesystems.jl")
include("commontransformations.jl")

end # module
