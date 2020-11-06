#############################
### 2D Coordinate systems ###
#############################
"""
`Polar{T,A}(r::T, θ::A)` - 2D polar coordinates
"""
struct Polar{T,A}
    r::T
    θ::A

    Polar{T, A}(r, θ) where {T, A} = new(r, θ)
end

function Polar(r, θ)
    r2, θ2 = promote(r, θ)

    return Polar{typeof(r2), typeof(θ2)}(r2, θ2)
end

Base.show(io::IO, x::Polar) = print(io, "Polar(r=$(x.r), θ=$(x.θ) rad)")
Base.isapprox(p1::Polar, p2::Polar; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...)

"`PolarFromCartesian()` - transformation from `AbstractVector` of length 2 to `Polar` type"
struct PolarFromCartesian <: Transformation; end
"`CartesianFromPolar()` - transformation from `Polar` type to `SVector{2}` type"
struct CartesianFromPolar <: Transformation; end

Base.show(io::IO, trans::PolarFromCartesian) = print(io, "PolarFromCartesian()")
Base.show(io::IO, trans::CartesianFromPolar) = print(io, "CartesianFromPolar()")

function (::PolarFromCartesian)(x::AbstractVector)
    length(x) == 2 || error("Polar transform takes a 2D coordinate")

    Polar(hypot(x[1], x[2]), atan(x[2], x[1]))
end

function transform_deriv(::PolarFromCartesian, x::AbstractVector)
    length(x) == 2 || error("Polar transform takes a 2D coordinate")

    r = hypot(x[1], x[2])
    f = x[2] / x[1]
    c = one(eltype(x))/(x[1]*(one(eltype(x)) + f*f))
    @SMatrix [ x[1]/r    x[2]/r ;
              -f*c       c      ]
end
transform_deriv_params(::PolarFromCartesian, x::AbstractVector) = error("PolarFromCartesian has no parameters")

function (::CartesianFromPolar)(x::Polar)
    s,c = sincos(x.θ)
    SVector(x.r * c, x.r * s)
end
function transform_deriv(::CartesianFromPolar, x::Polar)
    sθ, cθ = sincos(x.θ)
    @SMatrix [cθ  -x.r*sθ ;
              sθ   x.r*cθ ]
end
transform_deriv_params(::CartesianFromPolar, x::Polar) = error("CartesianFromPolar has no parameters")

Base.inv(::PolarFromCartesian) = CartesianFromPolar()
Base.inv(::CartesianFromPolar) = PolarFromCartesian()

compose(::PolarFromCartesian, ::CartesianFromPolar) = IdentityTransformation()
compose(::CartesianFromPolar, ::PolarFromCartesian) = IdentityTransformation()

# For convenience
Base.convert(::Type{Polar}, v::AbstractVector) = PolarFromCartesian()(v)
@inline Base.convert(::Type{V}, p::Polar) where {V <: AbstractVector} = convert(V, CartesianFromPolar()(p))
@inline Base.convert(::Type{V}, p::Polar) where {V <: StaticVector} = convert(V, CartesianFromPolar()(p))


#############################
### 3D Coordinate Systems ###
#############################
"""
Spherical(r, θ, ϕ) - 3D spherical coordinates

There are many Spherical coordinate conventions and this library uses a somewhat exotic one.
Given a vector `v` with Cartesian coordinates `xyz`, let `v_xy = [x,y,0]` be the orthogonal projection of `v` on the `xy` plane.

* `r` is the radius. It is given by `norm(v, 2)`.
* `θ` is the azimuth. It is the angle from the x-axis to `v_xy`
* `ϕ` is the latitude. It is the angle from `v_xy` to `v`.

```jldoctest
julia> using CoordinateTransformations

julia> v = randn(3);

julia> sph = SphericalFromCartesian()(v);

julia> r = sph.r; θ=sph.θ; ϕ=sph.ϕ;

julia> v ≈ [r * cos(θ) * cos(ϕ), r * sin(θ) * cos(ϕ), r*sin(ϕ)]
true
"""
struct Spherical{T,A}
    r::T
    θ::A
    ϕ::A

    Spherical{T, A}(r, θ, ϕ) where {T, A} = new(r, θ, ϕ)
end

function Spherical(r, θ, ϕ)
    r2, θ2, ϕ2 = promote(r, θ, ϕ)

    return Spherical{typeof(r2), typeof(θ2)}(r2, θ2, ϕ2)
end

Base.show(io::IO, x::Spherical) = print(io, "Spherical(r=$(x.r), θ=$(x.θ) rad, ϕ=$(x.ϕ) rad)")
Base.isapprox(p1::Spherical, p2::Spherical; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.ϕ, p2.ϕ; kwargs...)

"""
Cylindrical(r, θ, z) - 3D cylindrical coordinates
"""
struct Cylindrical{T,A}
    r::T
    θ::A
    z::T

    Cylindrical{T, A}(r, θ, z) where {T, A} = new(r, θ, z)
end

function Cylindrical(r, θ, z)
    r2, θ2, z2 = promote(r, θ, z)

    return Cylindrical{typeof(r2), typeof(θ2)}(r2, θ2, z2)
end

Base.show(io::IO, x::Cylindrical) = print(io, "Cylindrical(r=$(x.r), θ=$(x.θ) rad, z=$(x.z))")
Base.isapprox(p1::Cylindrical, p2::Cylindrical; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.z, p2.z; kwargs...)

"`SphericalFromCartesian()` - transformation from 3D point to `Spherical` type"
struct SphericalFromCartesian <: Transformation; end
"`CartesianFromSpherical()` - transformation from `Spherical` type to `SVector{3}` type"
struct CartesianFromSpherical <: Transformation; end
"`CylindricalFromCartesian()` - transformation from 3D point to `Cylindrical` type"
struct CylindricalFromCartesian <: Transformation; end
"`CartesianFromCylindrical()` - transformation from `Cylindrical` type to `SVector{3}` type"
struct CartesianFromCylindrical <: Transformation; end
"`CylindricalFromSpherical()` - transformation from `Spherical` type to `Cylindrical` type"
struct CylindricalFromSpherical <: Transformation; end
"`SphericalFromCylindrical()` - transformation from `Cylindrical` type to `Spherical` type"
struct SphericalFromCylindrical <: Transformation; end

Base.show(io::IO, trans::SphericalFromCartesian) = print(io, "SphericalFromCartesian()")
Base.show(io::IO, trans::CartesianFromSpherical) = print(io, "CartesianFromSpherical()")
Base.show(io::IO, trans::CylindricalFromCartesian) = print(io, "CylindricalFromCartesian()")
Base.show(io::IO, trans::CartesianFromCylindrical) = print(io, "CartesianFromCylindrical()")
Base.show(io::IO, trans::CylindricalFromSpherical) = print(io, "CylindricalFromSpherical()")
Base.show(io::IO, trans::SphericalFromCylindrical) = print(io, "SphericalFromCylindrical()")

# Cartesian <-> Spherical
function (::SphericalFromCartesian)(x::AbstractVector)
    length(x) == 3 || error("Spherical transform takes a 3D coordinate")

    Spherical(hypot(x[1], x[2], x[3]), atan(x[2], x[1]), atan(x[3], hypot(x[1], x[2])))
end
function transform_deriv(::SphericalFromCartesian, x::AbstractVector)
    length(x) == 3 || error("Spherical transform takes a 3D coordinate")
    T = eltype(x)

    r = hypot(x[1], x[2], x[3])
    rxy = hypot(x[1], x[2])
    fxy = x[2] / x[1]
    cxy = one(T)/(x[1]*(one(T) + fxy*fxy))
    f = -x[3]/(rxy*r*r)

    @SMatrix [ x[1]/r   x[2]/r  x[3]/r;
          -fxy*cxy  cxy     zero(T);
           f*x[1]   f*x[2]  rxy/(r*r) ]
end
transform_deriv_params(::SphericalFromCartesian, x::AbstractVector) = error("SphericalFromCartesian has no parameters")

function (::CartesianFromSpherical)(x::Spherical)
    sθ, cθ = sincos(x.θ)
    sϕ, cϕ = sincos(x.ϕ)
    SVector(x.r * cθ * cϕ, x.r * sθ * cϕ, x.r * sϕ)
end
function transform_deriv(::CartesianFromSpherical, x::Spherical{T}) where T
    sθ, cθ = sincos(x.θ)
    sϕ, cϕ = sincos(x.ϕ)
    @SMatrix [cθ*cϕ -x.r*sθ*cϕ -x.r*cθ*sϕ ;
              sθ*cϕ  x.r*cθ*cϕ -x.r*sθ*sϕ ;
              sϕ     zero(T)    x.r * cϕ  ]
end
transform_deriv_params(::CartesianFromSpherical, x::Spherical) = error("CartesianFromSpherical has no parameters")

# Cartesian <-> Cylindrical
function (::CylindricalFromCartesian)(x::AbstractVector)
    length(x) == 3 || error("Cylindrical transform takes a 3D coordinate")

    Cylindrical(hypot(x[1], x[2]), atan(x[2], x[1]), x[3])
end

function transform_deriv(::CylindricalFromCartesian, x::AbstractVector)
    length(x) == 3 || error("Cylindrical transform takes a 3D coordinate")
    T = eltype(x)

    r = hypot(x[1], x[2])
    f = x[2] / x[1]
    c = one(T)/(x[1]*(one(T) + f*f))
    @SMatrix [ x[1]/r   x[2]/r   zero(T) ;
              -f*c      c        zero(T) ;
               zero(T)  zero(T)  one(T)  ]
end
transform_deriv_params(::CylindricalFromCartesian, x::AbstractVector) = error("CylindricalFromCartesian has no parameters")

function (::CartesianFromCylindrical)(x::Cylindrical)
    sθ, cθ = sincos(x.θ)
    SVector(x.r * cθ, x.r * sθ, x.z)
end
function transform_deriv(::CartesianFromCylindrical, x::Cylindrical{T}) where {T}
    sθ, cθ = sincos(x.θ)
    @SMatrix [cθ      -x.r*sθ  zero(T) ;
              sθ       x.r*cθ  zero(T) ;
              zero(T)  zero(T) one(T)  ]
end
transform_deriv_params(::CartesianFromPolar, x::Cylindrical) = error("CartesianFromCylindrical has no parameters")

# Spherical <-> Cylindrical (TODO direct would be faster)
function (::CylindricalFromSpherical)(x::Spherical)
    CylindricalFromCartesian()(CartesianFromSpherical()(x))
end
function transform_deriv(::CylindricalFromSpherical, x::Spherical)
    M1 = transform_deriv(CylindricalFromCartesian(), CartesianFromSpherical()(x))
    M2 = transform_deriv(CartesianFromSpherical(), x)
    return M1*M2
end
transform_deriv_params(::CylindricalFromSpherical, x::Spherical) = error("CylindricalFromSpherical has no parameters")

function (::SphericalFromCylindrical)(x::Cylindrical)
    SphericalFromCartesian()(CartesianFromCylindrical()(x))
end
function transform_deriv(::SphericalFromCylindrical, x::Cylindrical)
    M1 = transform_deriv(SphericalFromCartesian(), CartesianFromCylindrical()(x))
    M2 = transform_deriv(CartesianFromCylindrical(), x)
    return M1*M2
end
transform_deriv_params(::SphericalFromCylindrical, x::Cylindrical) = error("SphericalFromCylindrical has no parameters")

Base.inv(::SphericalFromCartesian)   = CartesianFromSpherical()
Base.inv(::CartesianFromSpherical)   = SphericalFromCartesian()
Base.inv(::CylindricalFromCartesian) = CartesianFromCylindrical()
Base.inv(::CartesianFromCylindrical) = CylindricalFromCartesian()
Base.inv(::CylindricalFromSpherical) = SphericalFromCylindrical()
Base.inv(::SphericalFromCylindrical) = CylindricalFromSpherical()

# Inverse composition
compose(::SphericalFromCartesian,   ::CartesianFromSpherical)   = IdentityTransformation()
compose(::CartesianFromSpherical,   ::SphericalFromCartesian)   = IdentityTransformation()
compose(::CylindricalFromCartesian, ::CartesianFromCylindrical) = IdentityTransformation()
compose(::CartesianFromCylindrical, ::CylindricalFromCartesian) = IdentityTransformation()
compose(::CylindricalFromSpherical, ::SphericalFromCylindrical) = IdentityTransformation()
compose(::SphericalFromCylindrical, ::CylindricalFromSpherical) = IdentityTransformation()

# Cyclic compositions
compose(::SphericalFromCartesian,   ::CartesianFromCylindrical) = SphericalFromCylindrical()
compose(::CartesianFromSpherical,   ::SphericalFromCylindrical) = CartesianFromCylindrical()
compose(::CylindricalFromCartesian, ::CartesianFromSpherical)   = CylindricalFromSpherical()
compose(::CartesianFromCylindrical, ::CylindricalFromSpherical) = CartesianFromSpherical()
compose(::CylindricalFromSpherical, ::SphericalFromCartesian)   = CylindricalFromCartesian()
compose(::SphericalFromCylindrical, ::CylindricalFromCartesian) = SphericalFromCartesian()

# For convenience
Base.convert(::Type{Spherical}, v::AbstractVector) = SphericalFromCartesian()(v)
Base.convert(::Type{Cylindrical}, v::AbstractVector) = CylindricalFromCartesian()(v)

Base.convert(::Type{V}, s::Spherical) where {V <: AbstractVector} = convert(V, CartesianFromSpherical()(s))
Base.convert(::Type{V}, c::Cylindrical) where {V <: AbstractVector} = convert(V, CartesianFromCylindrical()(c))
Base.convert(::Type{V}, s::Spherical) where {V <: StaticVector} = convert(V, CartesianFromSpherical()(s))
Base.convert(::Type{V}, c::Cylindrical) where {V <: StaticVector} = convert(V, CartesianFromCylindrical()(c))

Base.convert(::Type{Spherical}, c::Cylindrical) = SphericalFromCylindrical()(c)
Base.convert(::Type{Cylindrical}, s::Spherical) = CylindricalFromSpherical()(s)
