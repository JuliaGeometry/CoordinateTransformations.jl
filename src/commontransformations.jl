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
immutable Rotation2D{T} <: AbstractTransformation{FixedVector{2}, FixedVector{2}}
    angle::T
    sin::T
    cos::T
end

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

# TODO
