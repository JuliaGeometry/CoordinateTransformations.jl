module CoordinateTransformations

using Compat
using FixedSizeArrays
export Point # Use Point{N, T} from FixedSizedArrays for Cartesian frames

# TODO: move these over to FixedSizeArrays at some point
function Base.vcat(v1::Vec, v2::Vec)
    (v1p, v2p) = promote(v1, v2)
    Vec(Tuple(v1p)..., Tuple(v2p)...)
end
function Base.hcat{N,M,P}(m1::Mat{N,M}, m2::Mat{N,P})
    (m1p, m2p) = promote(m1, m2)
    Mat(Tuple(m1p)..., Tuple(m2p)...)
end
@generated function Base.vcat{N,M,P}(m1::Mat{M,N}, m2::Mat{P,N})
    exprs = ntuple(i -> :( (Tuple(m1p)[$i]..., Tuple(m2p)[$i]...) ) , N)
    expr = Expr(:tuple, exprs...)
    quote
        (m1p, m2p) = promote(m1, m2)
        Mat($expr)
    end
end

using Rotations
export RotMatrix, Quaternion, SpQuat, AngleAxis, EulerAngles, ProperEulerAngles

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
export AbstractAffineTransformation
export AffineTransformation, LinearTransformation, Translation
export affine_decomposition_T_of_L, affine_decomposition_L_of_T
export RotationPolar, Rotation2D
export Rotation, RotationXY, RotationYZ, RotationZX
export RotationYX, RotationZY, RotationXZ, euler_rotation


include("core.jl")
include("coordinatesystems.jl")
include("commontransformations.jl")

# Deprecations
export transform
Base.@deprecate_binding AbstractTransformation Transformation
Base.@deprecate transform(transformation::Transformation, x) transformation(x)

end # module
