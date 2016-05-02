# Some common transformations are defined here

###################
### Translation ###
###################

immutable Translation{N, T} <: AbstractTransformation{FixedVector{N}, FixedVector{N}}
    dx::Vec{N, T}
end
Translation(x,y) = Translation(Vec(x,y))
Translation(x,y,z) = Translation(Vec(x,y,z))
Base.show(io::IO, trans::Translation) = print(io, "Translation$(trans.dx._)")

function transform{N}(trans::Translation{N}, x::FixedVector{N})
    (x_promoted, dx_promoted) = promote(x, trans.dx) # Force same data type
    x_promoted + (typeof(x_promoted))(dx_promoted) # Force same base type
end

Base.inv(trans::Translation) = Translation(-trans.dx)

function compose{N}(trans1::Translation{N}, trans2::Translation{N})
    Translation(trans1.dx + trans2.dx)
end

function transform_deriv{N}(trans::Translation{N}, x::Point{N})
    I
end

function transform_deriv_params{N}(trans::Translation{N}, x::Point{N})
    I
end

####################
### 2D Rotations ###
####################

# In polar coordinates
immutable RotationPolar{T} <: AbstractTransformation{Polar, Polar}
    angle::T
end
Base.show(io::IO, r::RotationPolar) = print(io, "RotationPolar($(r.angle))")


function transform(trans::RotationPolar, x::Polar)
    Polar(x.r, x.θ + trans.angle)
end

function transform_deriv{T}(trans::RotationPolar, x::Polar{T})
    @fsa [ zero(T) zero(T);
           zero(T) one(T)  ]
end

function transform_deriv_params{T}(trans::RotationPolar, x::Polar{T})
    @fsa [ zero(T);
           one(T)  ]
end

Base.inv(trans::RotationPolar) = RotationPolar(-trans.angle)

compose(t1::RotationPolar, t2::RotationPolar) = RotationPolar(t1.angle + t2.angle)

# In Cartesian coordinates
immutable Rotation2D{T} <: AbstractTransformation{FixedVector, FixedVector{2}}
    angle::T
    sin::T
    cos::T
end
Base.show(io::IO, r::Rotation2D) = print(io, "Rotation2D($(r.angle))")

function Rotation2D(a)
    s = sin(a)
    c = cos(a)
    return Rotation2D(a,s,c)
end

function transform(trans::Rotation2D, x::FixedVector{2})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[1]*sincos[2] - x[2]*sincos[1], x[1]*sincos[1] + x[2]*sincos[2])
end

function transform_deriv(trans::Rotation2D, x::FixedVector{2})
    @fsa [ trans.cos -trans.sin;
           trans.sin  trans.cos ]
end

function transform_deriv_params(trans::Rotation2D, x::FixedVector{2})
    # 2x1 transformation matrix
    Mat(-trans.sin*x[1] - trans.cos*x[2],
         trans.cos*x[1] - trans.sin*x[2] )
end

Base.inv(trans::Rotation2D) = Rotation2D(-trans.angle, -trans.sin, trans.cos)

compose(t1::Rotation2D, t2::Rotation2D) = Rotation2D(t1.angle + t2.angle)

##########################
### EulerRotation (3D) ###
##########################

immutable Rotation{R, T} <: AbstractTransformation{FixedVector{3}, FixedVector{3}}
    rotation::R
    matrix::Mat{3,3,T}
end
Base.show(io::IO, r::Rotation) = print(io, "Rotation($(r.rotation))")
Base.show(io::IO, r::Rotation{Void}) = print(io, "Rotation($(r.matrix))")


Rotation{T}(r::RotMatrix{T}) = Rotation(nothing, r) # R=Void represents direct parameterization by the marix (assumed to be Hermitian)
Rotation{T}(r::Matrix{T}) = Rotation(nothing, Mat{3,3,T}(r))
Rotation{T}(r::Quaternion{T}) = Rotation(r, convert_rotation(RotMatrix{T}, r))
Rotation{T}(r::SpQuat{T}) = Rotation(r, convert_rotation(RotMatrix{T}, r))
Rotation{T}(r::AngleAxis{T}) = Rotation(r, convert_rotation(RotMatrix{T}, r))
Rotation{Order,T}(r::EulerAngles{Order,T}) = Rotation(r, convert_rotation(RotMatrix{T}, r))
Rotation{Order,T}(r::ProperEulerAngles{Order,T}) = Rotation(r, convert_rotation(RotMatrix{T}, r))

import Base.==
==(a::Rotation, b::Rotation; kwargs...) = a.matrix == b.matrix
=={T}(a::Rotation{T},b::Rotation{T}) = a.matrix == b.matrix && a.rotation == b.rotation
Base.isapprox(a::Rotation, b::Rotation; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...)
Base.isapprox{T}(a::Rotation{T}, b::Rotation{T}; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...) && isapprox(a.rotation, b.rotation; kwargs...)
Base.isapprox(a::Rotation{Void}, b::Rotation{Void}; kwargs...) = isapprox(a.matrix, b.matrix; kwargs...)


function transform(trans::Rotation, x::FixedVector{3})
    (m, x2) = promote(trans.matrix, x)
    (typeof(x2))(m * Vec(x2))
end

transform_deriv(trans::Rotation, x::FixedVector{3}) = trans.matrix # It's a linear transformation, so this is easy!

function transform_deriv_params{T1,T2}(trans::Rotation{Void,T1}, x::FixedVector{3,T2})
    # This derivative isn't projected into the orthogonal/Hermition tangent
    # plane. It would be acheived by:
    # Δ -> (Δ - R Δ' R) / 2

    # The matrix gives 9 parameters...
    Z = zero(promote_type(T1,T2))
    @fsa [ x[1] x[2] x[3] Z Z Z Z Z Z;
           Z Z Z x[1] x[2] x[3] Z Z Z;
           Z Z Z Z Z Z x[1] x[2] x[3] ]
end


function transform_deriv_params{T1,T2}(trans::Rotation{Quaternion{T1},T1}, x::FixedVector{3,T2})
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
immutable RotationXY{T} <: AbstractTransformation{FixedVector{3}, FixedVector{3}}
    angle::T
    sin::T
    cos::T
end
immutable RotationYZ{T} <: AbstractTransformation{FixedVector{3}, FixedVector{3}}
    angle::T
    sin::T
    cos::T
end
immutable RotationZX{T} <: AbstractTransformation{FixedVector{3}, FixedVector{3}}
    angle::T
    sin::T
    cos::T
end

function RotationXY(a)
    s = sin(a)
    c = cos(a)
    return RotationXY(a,s,c)
end
RotationYX(a) = RotationXY(-a)
function RotationYZ(a)
    s = sin(a)
    c = cos(a)
    return RotationYZ(a,s,c)
end
RotationZY(a) = RotationYZ(-a)
function RotationZX(a)
    s = sin(a)
    c = cos(a)
    return RotationZX(a,s,c)
end
RotationXZ(a) = RotationZX(-a)

Base.show(io::IO, r::RotationXY) = print(io, "RotationXY($(r.angle))")
Base.show(io::IO, r::RotationYZ) = print(io, "RotationYZ($(r.angle))")
Base.show(io::IO, r::RotationZX) = print(io, "RotationZX($(r.angle))")

function transform(trans::RotationXY, x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[1]*sincos[2] - x[2]*sincos[1], x[1]*sincos[1] + x[2]*sincos[2], x2[3])
end
function transform(trans::RotationYZ, x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x2[1], x[2]*sincos[2] - x[3]*sincos[1], x[2]*sincos[1] + x[3]*sincos[2])
end
function transform(trans::RotationZX, x::FixedVector{3})
    (sincos, x2) = promote(Vec(trans.sin, trans.cos), x)
    (typeof(x2))(x[3]*sincos[1] + x[1]*sincos[2], x2[2], x[3]*sincos[2] - x[1]*sincos[1])
end

function transform_deriv(trans::RotationXY, x::FixedVector{3})
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos -trans.sin Z;
           trans.sin  trans.cos Z;
           Z          Z         I]
end
function transform_deriv(trans::RotationYZ, x::FixedVector{3})
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ I  Z         Z;
           Z  trans.cos -trans.sin;
           Z  trans.sin  trans.cos ]
end
function transform_deriv(trans::RotationZX, x::FixedVector{3})
    Z = zero(trans.cos)
    I = one(trans.cos)
    @fsa [ trans.cos Z trans.sin;
           Z         I Z        ;
          -trans.sin Z trans.cos ]
end

function transform_deriv_params(trans::RotationXY, x::FixedVector{3})
    # 2x1 transformation matrix
    Z = zero(promote_type(typeof(trans.cos), eltype(x)))
    Mat(-trans.sin*x[1] - trans.cos*x[2],
         trans.cos*x[1] - trans.sin*x[2],
         Z)
end
function transform_deriv_params(trans::RotationYZ, x::FixedVector{3})
    # 2x1 transformation matrix
    Z = zero(promote_type(typeof(trans.cos), eltype(x)))
    Mat( Z,
        -trans.sin*x[2] - trans.cos*x[3],
         trans.cos*x[2] - trans.sin*x[3])
end
function transform_deriv_params(trans::RotationZX, x::FixedVector{3})
    # 2x1 transformation matrix
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
