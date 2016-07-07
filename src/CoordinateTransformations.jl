module CoordinateTransformations

using FixedSizeArrays
export Point # Use Point{N, T} from FixedSizedArrays for Cartesian frames

# TODO: move these over to FixedSizeArrays at some point
function Base.vcat(v1::Vec, v2::Vec)
    (v1p, v2p) = promote(v1, v2)
    Vec(v1p._..., v2p._...)
end
function Base.hcat{N,M,P}(m1::Mat{N,M}, m2::Mat{N,P})
    (m1p, m2p) = promote(m1, m2)
    Mat(m1p._..., m2p._...)
end
@generated function Base.vcat{N,M,P}(m1::Mat{M,N}, m2::Mat{P,N})
    exprs = ntuple(i -> :( (m1p._[$i]..., m2p._[$i]...) ) , N)
    expr = Expr(:tuple, exprs...)
    quote
        (m1p, m2p) = promote(m1, m2)
        Mat($expr)
    end
end

using Rotations
export RotMatrix, Quaternion, SpQuat, AngleAxis, EulerAngles, ProperEulerAngles

# Core methods
export transform, compose, ∘, transform_deriv, transform_deriv_params
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
export Translation
export RotationPolar, Rotation2D
export Rotation, RotationXY, RotationYZ, RotationZX
export RotationYX, RotationZY, RotationXZ, euler_rotation

#export RigidBodyTransformation, AffineTransformation

include("core.jl")
include("coordinatesystems.jl")
include("commontransformations.jl")

end # module
