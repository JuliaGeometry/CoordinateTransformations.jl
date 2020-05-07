module CoordinateTransformations

using StaticArrays
using LinearAlgebra

using Rotations

# Re-export useful rotation types from Rotations.jl
export RotMatrix, AngleAxis,
       RotX, RotY, RotZ,
       RotXY, RotYX, RotZX, RotXZ, RotYZ, RotZY,
       RotXYX, RotYXY, RotZXZ, RotXZX, RotYZY, RotZYZ,
       RotXYZ, RotYXZ, RotZXY, RotXZY, RotYZX, RotZYX

# Export quaternion and rotation vector types for Rotations.jl < v1.0
# Note: it's ok to have these `export` commands even for later versions
# of Rotations.jl`. Doing `export foo` is allowed when `foo` is undefined.
export Quat, SpQuat, RodriguesVec

# Export quaternion and rotation vector types for Rotations.jl 1.0
export UnitQuaternion, MRP, RotationVector

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

include("core.jl")
include("coordinatesystems.jl")
include("affine.jl")
include("perspective.jl")

# Deprecations
export transform
Base.@deprecate_binding AbstractTransformation Transformation
Base.@deprecate transform(transformation::Transformation, x) transformation(x)

end # module
