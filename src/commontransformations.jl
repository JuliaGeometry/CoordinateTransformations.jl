"""
    abstract AbstractAffineTransformation <: Transformation

Provides an interface for implementing Affine transformations of Cartesian
coordinates. To implement an AbstractAffineTransformation, you must define

    matrix(trans)
    translation(trans)

where the resulting transformation is (equivalent to)

    trans(x) -> matrix(trans) * x + translation(trans)

Specific implementations may provide equivalent specializations of `call`, etc,
for optimization purposes. The function `translation_reverse()` is provided,
such that

    trans(x) -> matrix(trans) * (x + translation_reverse(trans))

(See also AffineTransformation, AbstractLinearTransformation, Translation)
"""
abstract AbstractAffineTransformation <: Transformation

matrix(::AbstractAffineTransformation) = error("AbstractAffineTransformation's must implement matrix()")
translation(::AbstractAffineTransformation) = error("AbstractAffineTransformation's must implement translation()")
translation_reverse(::AbstractAffineTransformation) = matrix(trans) \ translation(trans)

# Default implementations
@compat function (trans::AbstractAffineTransformation)(x)
    matrix(trans) * x + translation(trans)
end

# Could try do similar for transform_deriv_params()?

transform_deriv(trans::AbstractAffineTransformation, x) = matrix(trans)

function Base.inv(trans::AbstractAffineTransformation)
    Minv = inv(matrix(trans))
    AffineTransformation(Minv, -Minv * translation(trans))
end

function compose(t1::AbstractAffineTransformation, t2::AbstractAffineTransformation)
    AffineTransformation(matrix(t1) * matrix(t2), translation(t1) + matrix(t1) * translation(t2))
end

function Base.isapprox(t1::AbstractAffineTransformation, t2::AbstractAffineTransformation, kwargs...)
    isapprox(matrix(t1), matrix(t2); kwargs...) &&
        isapprox(translation(t1), translation(t2); kwargs...)
end

"""
    abstract AbstractLinearTransformation <: AbstractAffineTransformation

Provides an interface for implementing linear transformations of Cartesian
coordinates. To implement an AbstractLinearTransformation, you must define

    matrix(trans)

where the resulting transformation is (equivalent to)

    trans(x) -> matrix(trans) * x

Specific implementations may provide equivalent specializations of `call`, etc,
for optimization purposes.

(See also LinearTransformation, AbstractAffineTransformation)
"""
abstract AbstractLinearTransformation <: AbstractAffineTransformation

matrix(::AbstractLinearTransformation) = error("AbstractLinearTransformation's must implement matrix()")
function translation(trans::AbstractLinearTransformation)
    m = matrix(trans)
    s = size(m, 1)
    T = eltype(m)
    return zeros(T, s)
end
function translation_reverse(trans::AbstractLinearTransformation)
    m = matrix(trans)
    s = size(m, 2)
    T = eltype(m)
    return zeros(T, s)
end

# Default implementations
@compat function (trans::AbstractLinearTransformation)(x)
    matrix(trans) * x
end

# transform_deriv() identical to that provided by AbstractAffineTransformation

Base.inv(trans::AbstractLinearTransformation) = LinearTransformation(inv(matrix(trans)))

compose(t1::AbstractLinearTransformation, t2::AbstractLinearTransformation) = LinearTransformation(matrix(t1) * matrix(t2))

function Base.isapprox(t1::AbstractLinearTransformation, t2::AbstractLinearTransformation, kwargs...)
    isapprox(matrix(t1), matrix(t2); kwargs...)
end



# These functions are remove any reference to unnecessary calls to
# translation(::LinearTransformation) from the AbstractAffineTransformations
# interface:

function compose(t1::AbstractAffineTransformation, t2::AbstractLinearTransformation)
    AffineTransformation(matrix(t1) * matrix(t2), translation(t1))
end

function compose(t1::AbstractLinearTransformation, t2::AbstractAffineTransformation)
    AffineTransformation(matrix(t1) * matrix(t2), matrix(t1) * translation(t2))
end

function Base.isapprox(t1::AbstractAffineTransformation, t2::AbstractLinearTransformation, kwargs...)
    isapprox(matrix(t1), matrix(t2); kwargs...) &&
        isapprox(norm(translation(t1)), 0; kwargs...)
end

function Base.isapprox(t1::AbstractLinearTransformation, t2::AbstractAffineTransformation, kwargs...)
    isapprox(matrix(t1), matrix(t2); kwargs...) &&
        isapprox(norm(translation(t2)), 0; kwargs...)
end

"""
    LinearTransformation <: AbstractLinearTransformation

A general linear transformation, constructed using `LinearTransformation(M)`
for any matrix-like object `M`.  Other abstract linear transformations can be
converted into a general linear transformation using `LinearTransformation(trans)`
"""
immutable LinearTransformation{MatrixT} <: AbstractLinearTransformation
    M::MatrixT
end

LinearTransformation(trans::AbstractLinearTransformation) = LinearTransformation(matrix(trans))

Base.show(io::IO, trans::LinearTransformation)   = print(io, "LinearTransformation($(trans.M))") # TODO make this output more petite

@inline matrix(trans::LinearTransformation) = trans.M


"""
    AffineTransformation <: AbstractAffineTransformation

A concrete affine transformation.  To construct the mapping `x -> M*x + v`, use

    AffineTransformation(M, v)

where `M` is a matrix and `v` a vector.  An arbitrary `Transformation` may be
converted into an affine approximation by linearizing about a point `x` using

    AffineTransformation(trans, [x])

For transformations which are already affine, `x` may be omitted.
"""
immutable AffineTransformation{MatrixT, VectorT} <: AbstractAffineTransformation
    M::MatrixT
    v::VectorT
end

matrix(trans::AffineTransformation)      = trans.M
translation(trans::AffineTransformation) = trans.v

function AffineTransformation(trans::AbstractAffineTransformation)
    AffineTransformation(matrix(trans), translation(trans))
end

# We can create an Affine transformation corresponding to the differential
# transformation of x + dx
#
# Note: the expression `Tx - dT*Tx` will have large cancellation error for
# large Tx!  However, changing the order of applying the matrix and
# translation won't fix things, because then we'd have `Tx*(x-x0)` which
# also can incur large cancellation error in `x-x0`.
function AffineTransformation(trans::Transformation, x0)
    dT = transform_deriv(trans, x0)
    Tx = trans(x0)
    AffineTransformation(dT, Tx - dT*x0)
end

Base.show(io::IO, trans::AffineTransformation) = print(io, "AffineTransformation($(trans.M), $(trans.v))") # TODO make this output more petite



###################
### Translation ###
###################
"""
    Translation(dv) <: AbstractAffineTransformation
    Translation(dx, dy)       (2D)
    Translation(dx, dy, dz)   (3D)

Construct the `Translation` transformation for translating Cartesian points.
"""
immutable Translation{T} <: AbstractAffineTransformation
    dx::T
end
Translation(x::Tuple) = Translation(Vec(x))
Translation(x,y) = Translation(Vec(x,y))
Translation(x,y,z) = Translation(Vec(x,y,z))
Base.show(io::IO, trans::Translation) = print(io, "Translation$((trans.dx...))")

@inline matrix(::Translation) = I
@inline translation(trans::Translation) = trans.dx
@inline translation(trans::Translation) = trans.dx

@compat function (trans::Translation)(x)
    x + trans.dx
end

@compat (trans::Translation)(x::Tuple) = Tuple(Vec(x) + trans.dx)

Base.inv(trans::Translation) = Translation(-trans.dx)

function compose(trans1::Translation, trans2::Translation)
    Translation(trans1.dx + trans2.dx)
end

function transform_deriv_params(trans::Translation, x)
    I
end

####################
### 2D Rotations ###
####################

"""
    Rotation2D(angle)

Construct the `Rotation2D` transformation for rotating 2D Cartesian points
(i.e. `FixedVector{2}`s) about the origin.
"""
immutable Rotation2D{T} <: AbstractLinearTransformation
    angle::T
    sin::T
    cos::T
end
Base.show(io::IO, r::Rotation2D) = print(io, "Rotation2D($(r.angle) rad)")

function Rotation2D(a)
    s = sin(a)
    c = cos(a)
    return Rotation2D(promote(a,s,c)...)
end

# A variety of specializations for all occassions!
@compat function (trans::Rotation2D)(x::FixedVector{2})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[1]*sincos[2] - x[2]*sincos[1], x[1]*sincos[1] + x[2]*sincos[2])
end

@compat (trans::Rotation2D)(x::NTuple{2})             = (x[1]*trans.cos - x[2]*trans.sin, x[1]*trans.sin + x[2]*trans.cos)
@compat (trans::Rotation2D{T1}){T1,T2}(x::Vector{T2}) = [x[1]*trans.cos - x[2]*trans.sin, x[1]*trans.sin + x[2]*trans.cos]

# E.g. for ArrayFire, this might work better than the below?
@compat function (trans::Rotation2D{T1}){T1,T2}(x::AbstractVector{T2})
    out = similar(x, promote_type(T1,T2))
    out[1] = x[1]*trans.cos - x[2]*trans.sin
    out[2] = x[1]*trans.sin + x[2]*trans.cos
    return out
end

@compat function (trans::Rotation2D)(x)
    [ trans.cos -trans.sin;
      trans.sin  trans.cos ] * x
end

@compat function (trans::Rotation2D)(x::Polar)
    Polar(x.r, x.θ + trans.angle)
end

function matrix(trans::Rotation2D)
    @fsa [ trans.cos -trans.sin;
           trans.sin  trans.cos ]
end

function transform_deriv(trans::Rotation2D, x)
    @fsa [ trans.cos -trans.sin;
           trans.sin  trans.cos ]
end

function transform_deriv{T}(trans::Rotation2D, x::Polar{T})
    @fsa [ zero(T) zero(T);
           zero(T) one(T)  ]
end

function transform_deriv_params(trans::Rotation2D, x)
    # 2x1 transformation matrix
    Mat(-trans.sin*x[1] - trans.cos*x[2],
         trans.cos*x[1] - trans.sin*x[2] )
end

function transform_deriv_params{T}(trans::Rotation2D, x::Polar{T})
    @fsa [ zero(T);
           one(T)  ]
end

Base.inv(trans::Rotation2D) = Rotation2D(-trans.angle, -trans.sin, trans.cos)

compose(t1::Rotation2D, t2::Rotation2D) = Rotation2D(t1.angle + t2.angle)




#####################
### Rotation (3D) ###
#####################
"""
    Rotation(R)

Construct the `Rotation` transformation for rotating 3D Cartesian points
(i.e. `FixedVector{3}`s) about the origin. I
"""
immutable Rotation{R, T} <: AbstractLinearTransformation
    rotation::R
    matrix::Mat{3,3,T} # Should we enforce this storage, or merely "suggest" it
end
Base.show(io::IO, r::Rotation) = print(io, "Rotation($(r.rotation))")
Base.show(io::IO, r::Rotation{Void}) = print(io, "Rotation($(r.matrix))")

"""
    Rotation(matrix)

Construct the `Rotation` transformation for rotating 3D Cartesian points
(i.e. `FixedVector{3}`s) about the origin. `matrix` is a 3×3 `Matrix` or `Mat`,
and is assumed to be orthogonal.
"""
Rotation{T}(r::RotMatrix{T}) = Rotation(nothing, r) # R=Void represents direct parameterization by the marix (assumed to be orthogonal/unitary)
Rotation{T}(r::Matrix{T}) = Rotation(nothing, Mat{3,3,T}(r)) # Should we enforce this storage, or merely "suggest" it
"""
    Rotation(R)

Construct the `Rotation` transformation for rotating 3D Cartesian points
(i.e. `FixedVector{3}`s) about the origin. `R` is a rotation object defined
in the Rotations package (`Quaternion`, `SpQuat`, `AngleAxis`, `EulerAngles`
or `ProperEulerAngles`). From these a 3×3 rotation matrix is constructed and
cached, along with the original parameter specifications (which is used for
`transform_deriv_params`).

(see also `RotationXY`, `RotationYZ`, `RotationZX` and `euler_rotation`)
"""
Rotation{T}(r::Quaternion{T}) = Rotation(r, convert(RotMatrix{T}, r))
Rotation{T}(r::SpQuat{T}) = Rotation(r, convert(RotMatrix{T}, r))
Rotation{T}(r::AngleAxis{T}) = Rotation(r, convert(RotMatrix{T}, r))
Rotation{Order,T}(r::EulerAngles{Order,T}) = Rotation(r, convert(RotMatrix{T}, r))
Rotation{Order,T}(r::ProperEulerAngles{Order,T}) = Rotation(r, convert(RotMatrix{T}, r))

import Base.==
==(a::Rotation, b::Rotation; kwargs...) = a.matrix == b.matrix
=={T}(a::Rotation{T},b::Rotation{T}) = a.matrix == b.matrix && a.rotation == b.rotation
Base.isapprox(a::Rotation, b::Rotation; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...)
Base.isapprox{T}(a::Rotation{T}, b::Rotation{T}; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...) && isapprox(a.rotation, b.rotation; kwargs...)
Base.isapprox(a::Rotation{Void}, b::Rotation{Void}; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...)

@compat function (trans::Rotation)(x)
    trans.matrix * x
end

@compat function (trans::Rotation)(x::FixedVector{3})
    (m, x2) = promote(trans.matrix, x)
    (typeof(x2))(m * Vec(x2))
end

@compat (trans::Rotation)(x::Tuple) = Tuple(trans(Vec(x)))

transform_deriv(trans::Rotation, x) = trans.matrix # It's a linear transformation, so this is easy!
@inline matrix(trans::Rotation2D) = trans.matrix


function transform_deriv_params{T}(trans::Rotation{Void,T}, x)
    # This derivative isn't projected into the orthogonal/Hermition tangent
    # plane. It would be acheived by:
    # Δ -> (Δ - R Δ' R) / 2

    # The matrix gives 9 parameters...
    Z = zero(promote_type(T, eltype(x)))
    @fsa [ x[1] x[2] x[3] Z Z Z Z Z Z;
           Z Z Z x[1] x[2] x[3] Z Z Z;
           Z Z Z Z Z Z x[1] x[2] x[3] ]
end


function transform_deriv_params{T1,T2}(trans::Rotation{Quaternion{T1},T2}, x)
    #=
    # From Rotations.jl package
    # get rotation matrix from quaternion
    xx = q.v1 * q.v1
    yy = q.v2 * q.v2
    zz = q.v3 * q.v3
    xy = q.v1 * q.v2
    zw = q.s  * q.v3
    xz = q.v1 * q.v3
    yw = q.v2 * q.s
    yz = q.v2 * q.v3
    xw = q.s  * q.v1

    # initialize rotation part
    return @fsa([1 - 2 * (yy + zz)    2 * (xy - zw)       2 * (xz + yw);
                 2 * (xy + zw)        1 - 2 * (xx + zz)   2 * (yz - xw);
                 2 * (xz - yw)        2 * (yz + xw)       1 - 2 * (xx + yy)])

    # This leads to these derivative matrices:

    dR/ds =
    [ 0    -2*v3    2*v2;
     2*v3     0    -2*v1;
    -2*v2  2*v1     0   ]

    dR/dv1 =
    [ 0  2*v2  2*v3 ;
     2*v2 -4*v1 -2*s;
     2*v3  2*s  -4*v1]

    dR/dv2 =
    [-4*v2  2*v1  2*s;
      2*v1  0     2*v3;
     -2*s   2*v3  -4*v2]

    dR/dv3 =
    [-4*v3  -2*s  2*v1;
     2*s    -4*v3  2*v2;
     2*v1   2*v2   0 ]
    =#
    s = trans.rotation.s
    v1 = trans.rotation.v1
    v2 = trans.rotation.v2
    v3 = trans.rotation.v3

    @fsa [ 2(-v3*x[2]+v2*x[3])  2(v2*x[2]+v3*x[3])            2(-2*v2*x[1]+v1*x[2]+s*x[3])  2(-2*v3*x[1]-s*x[2]+v1*x[3]) ;
           2(v3*x[1]-v1*x[3])   2(v2*x[1]-2*v1*x[2]- s*x[3])  2(v1*x[1]+v3*x[3])            2(s*x[1]-2*v3*x[2]+v2*x[3])  ;
           2(-v2*x[1]+v1*x[2])  2(v3*x[1]+ s*x[2]-2*v1*x[3])  2(-s*x[1]+v3*x[2]-2*v2*x[3])  2(v1*x[1]+v2*x[2])           ]
end

#function transform_deriv_params{T1,T2}(trans::Rotation{EulerAngles{T1},T1}, x::FixedVector{3,T2})
#end

Base.inv(trans::Rotation) = Rotation(nothing, trans.matrix')
Base.inv{T <: Union{Quaternion, SpQuat}}(trans::Rotation{T}) = Rotation(inv(trans.rotation), trans.matrix')

compose(t1::Rotation, t2::Rotation) = Rotation(nothing, t1.matrix*t2.matrix) # A bit lossy for transform_deriv_params, but otherwise probably the most efficient choice


#############################
# Individual axis rotations #
#############################

"""
    RotationXY(angle)

Construct the `RotationXY` transformation for rotating 3D Cartesian points
(e.g. `Vector`, `NTuple{3}`, or `FixedVector{3}`) through the X-Y plane (around
the Z axis).

(see also `Rotation`, `RotationYZ`, `RotationZX` and `euler_rotation`)
"""
immutable RotationXY{T} <: AbstractLinearTransformation
    angle::T
    sin::T
    cos::T
end
"""
    RotationYZ(angle)

Construct the `RotationYZ` transformation for rotating 3D Cartesian points
(e.g. `Vector`, `NTuple{3}`, or `FixedVector{3}`) through the Y-Z plane (around
the X axis).

(see also `Rotation`, `RotationXY`, `RotationZX` and `euler_rotation`)
"""
immutable RotationYZ{T} <: AbstractLinearTransformation
    angle::T
    sin::T
    cos::T
end
"""
    RotationZX(angle)

Construct the `RotationZX` transformation for rotating 3D Cartesian points
(e.g. `Vector`, `NTuple{3}`, or `FixedVector{3}`) through the Z-X plane (around
the Y axis).

(see also `Rotation`, `RotationXY`, `RotationYZ` and `euler_rotation`)
"""
immutable RotationZX{T} <: AbstractLinearTransformation
    angle::T
    sin::T
    cos::T
end

function RotationXY(a)
    s = sin(a)
    c = cos(a)
    return RotationXY(promote(a,s,c)...)
end
"RotationYX(angle) - constructs RotationXY(-angle)"
RotationYX(a) = RotationXY(-a)
function RotationYZ(a)
    s = sin(a)
    c = cos(a)
    return RotationYZ(promote(a,s,c)...)
end
"RotationZY(angle) - constructs RotationYZ(-angle)"
RotationZY(a) = RotationYZ(-a)
function RotationZX(a)
    s = sin(a)
    c = cos(a)
    return RotationZX(promote(a,s,c)...)
end
"RotationXZ(angle) - constructs RotationZX(-angle)"
RotationXZ(a) = RotationZX(-a)

Base.show(io::IO, r::RotationXY) = print(io, "RotationXY($(r.angle))")
Base.show(io::IO, r::RotationYZ) = print(io, "RotationYZ($(r.angle))")
Base.show(io::IO, r::RotationZX) = print(io, "RotationZX($(r.angle))")

# It's a little fiddly to support all possible point container types, but this should do a good majority of them!
@compat (trans::RotationXY)(x::Vector) =    [x[1]*trans.cos - x[2]*trans.sin, x[1]*trans.sin + x[2]*trans.cos, x[3]]
@compat (trans::RotationXY)(x::NTuple{3}) = (x[1]*trans.cos - x[2]*trans.sin, x[1]*trans.sin + x[2]*trans.cos, x[3])
@compat function (trans::RotationXY)(x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[1]*sincos[2] - x[2]*sincos[1], x[1]*sincos[1] + x[2]*sincos[2], x2[3])
end
@compat function (trans::RotationXY{T}){T}(x)
    Z = zero(T)
    I = one(T)

    [trans.cos -trans.sin Z;
    trans.sin  trans.cos Z;
    Z          Z         I ] * x
end

@compat (trans::RotationYZ)(x::Vector) =    [x[1], x[2]*trans.cos - x[3]*trans.sin, x[2]*trans.sin + x[3]*trans.cos]
@compat (trans::RotationYZ)(x::NTuple{3}) = (x[1], x[2]*trans.cos - x[3]*trans.sin, x[2]*trans.sin + x[3]*trans.cos)
@compat function (trans::RotationYZ)(x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x2[1], x[2]*sincos[2] - x[3]*sincos[1], x[2]*sincos[1] + x[3]*sincos[2])
end
@compat function (trans::RotationYZ{T}){T}(x)
    Z = zero(T)
    I = one(T)

    [I Z          Z        ;
     Z trans.cos -trans.sin;
     Z trans.sin  trans.cos] * x
end

@compat (trans::RotationZX)(x::Vector) =    [x[3]*trans.sin + x[1]*trans.cos, x[2], x[3]*trans.cos - x[1]*trans.sin]
@compat (trans::RotationZX)(x::NTuple{3}) = (x[3]*trans.sin + x[1]*trans.cos, x[2], x[3]*trans.cos - x[1]*trans.sin)
@compat function (trans::RotationZX)(x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[3]*sincos[1] + x[1]*sincos[2], x2[2], x[3]*sincos[2] - x[1]*sincos[1])
end
@compat function (trans::RotationZX{T}){T}(x)
    Z = zero(T)
    I = one(T)

    [ trans.cos Z trans.sin;
      Z         I Z        ;
     -trans.sin Z trans.cos] * x
end

function transform_deriv(trans::RotationXY, x)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos -trans.sin Z;
           trans.sin  trans.cos Z;
           Z          Z         I]
end
function matrix(trans::RotationXY)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos -trans.sin Z;
           trans.sin  trans.cos Z;
           Z          Z         I]
end

function transform_deriv(trans::RotationYZ, x)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ I  Z         Z;
           Z  trans.cos -trans.sin;
           Z  trans.sin  trans.cos ]
end
function matrix(trans::RotationYZ)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ I  Z         Z;
           Z  trans.cos -trans.sin;
           Z  trans.sin  trans.cos ]
end

function transform_deriv(trans::RotationZX, x)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos Z trans.sin;
           Z         I Z        ;
          -trans.sin Z trans.cos ]
end
function matrix(trans::RotationZX)
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos Z trans.sin;
           Z         I Z        ;
          -trans.sin Z trans.cos ]
end

function transform_deriv_params(trans::RotationXY, x)
    # 3x1 transformation matrix
    Z = zero(promote_type(typeof(trans.cos), eltype(x)))
    Mat(-trans.sin*x[1] - trans.cos*x[2],
         trans.cos*x[1] - trans.sin*x[2],
         Z)
end
function transform_deriv_params(trans::RotationYZ, x)
    # 3x1 transformation matrix
    Z = zero(promote_type(typeof(trans.cos), eltype(x)))
    Mat( Z,
        -trans.sin*x[2] - trans.cos*x[3],
         trans.cos*x[2] - trans.sin*x[3])
end
function transform_deriv_params(trans::RotationZX, x)
    # 3x1 transformation matrix
    Z = zero(promote_type(typeof(trans.cos), eltype(x)))
    Mat( trans.cos*x[3] - trans.sin*x[1],
         Z,
        -trans.sin*x[3] - trans.cos*x[1])
end

Base.inv(trans::RotationXY) = RotationXY(-trans.angle, -trans.sin, trans.cos)
Base.inv(trans::RotationYZ) = RotationYZ(-trans.angle, -trans.sin, trans.cos)
Base.inv(trans::RotationZX) = RotationZX(-trans.angle, -trans.sin, trans.cos)

compose(t1::RotationXY, t2::RotationXY) = RotationXY(t1.angle + t2.angle)
compose(t1::RotationYZ, t2::RotationYZ) = RotationYZ(t1.angle + t2.angle)
compose(t1::RotationZX, t2::RotationZX) = RotationZX(t1.angle + t2.angle)

# defualts to EulerZXY
euler_rotation(θ₁, θ₂, θ₃) = euler_rotation(θ₁, θ₂, θ₃, Rotations.EulerZXY)

# Tait-Bryant orderings
"""
    euler_rotation(θ₁, θ₂, θ₃, [order = Rotations.EulerZXY])

Constructs a composed set of elementary (planar) rotations from the three given
Euler angles. `order` is defined in the Rotations package, and can be either
a Tait-Bryant ordering (ABC) or proper Euler ordering (ABA).

(see also `Rotation`, `RotationXY`, `RotationYZ` and `RotationZX`)
"""
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerZYX, Type{Rotations.EulerZYX}})
    RotationXY(θ₁) ∘ RotationZX(θ₂) ∘ RotationYZ(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerZXY, Type{Rotations.EulerZXY}})
    RotationXY(θ₁) ∘ RotationYZ(θ₂) ∘ RotationZX(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerYZX, Type{Rotations.EulerYZX}})
    RotationZX(θ₁) ∘ RotationXY(θ₂) ∘ RotationYZ(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerYXZ, Type{Rotations.EulerYXZ}})
    RotationZX(θ₁) ∘ RotationYZ(θ₂) ∘ RotationXY(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerXYZ, Type{Rotations.EulerXYZ}})
    RotationYZ(θ₁) ∘ RotationZX(θ₂) ∘ RotationXY(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerXZY, Type{Rotations.EulerXZY}})
    RotationYZ(θ₁) ∘ RotationXY(θ₂) ∘ RotationZX(θ₃)
end

# Proper Euler orderings
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerZYZ, Type{Rotations.EulerZYZ}})
    RotationXY(θ₁) ∘ RotationZX(θ₂) ∘ RotationXY(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerZXZ, Type{Rotations.EulerZXZ}})
    RotationXY(θ₁) ∘ RotationYZ(θ₂) ∘ RotationXY(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerYZY, Type{Rotations.EulerYZY}})
    RotationZX(θ₁) ∘ RotationXY(θ₂) ∘ RotationZX(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerYXY, Type{Rotations.EulerYXY}})
    RotationZX(θ₁) ∘ RotationYZ(θ₂) ∘ RotationZX(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerXYX, Type{Rotations.EulerXYX}})
    RotationYZ(θ₁) ∘ RotationZX(θ₂) ∘ RotationYZ(θ₃)
end
function euler_rotation(θ₁, θ₂, θ₃, order::Union{Rotations.EulerXZX, Type{Rotations.EulerXZX}})
    RotationYZ(θ₁) ∘ RotationXY(θ₂) ∘ RotationYZ(θ₃)
end
