module CoordinateTransformations

using StaticArrays

using Rotations
export RotMatrix, Quat, SpQuat, AngleAxis, RodriguesVec,
       RotX, RotY, RotZ,
       RotXY, RotYX, RotZX, RotXZ, RotYZ, RotZY,
       RotXYX, RotYXY, RotZXZ, RotXZX, RotYZY, RotZYZ,
       RotXYZ, RotYXZ, RotZXY, RotXZY, RotYZX, RotZYX

# Core methods
export compose, ∘, transform_deriv, transform_deriv_params
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
export AbstractAffineTransformation, AbstractLinearTransformation, AbstractTranslation
export AffineTransformation, LinearTransformation, Translation, transformation_matrix, translation_vector, translation_vector_reverse

include("core.jl")
include("coordinatesystems.jl")
include("affine.jl")

# Deprecations
export transform
Base.@deprecate_binding AbstractTransformation Transformation
Base.@deprecate transform(transformation::Transformation, x) transformation(x)

end # module
