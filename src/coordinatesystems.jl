#############################
### 2D Coordinate systems ###
#############################
immutable Polar{T}
    r::T
    θ::T
end
Base.show(io::IO, x::Polar) = print(io, "Polar(r=$(x.r), θ=$(x.θ) rad)")
Base.isapprox(p1::Polar, p2::Polar; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...)

immutable PolarFromCartesian <: AbstractTransformation{Polar, Point{2}}; end
immutable CartesianFromPolar <: AbstractTransformation{Point{2}, Polar}; end

Base.show(io::IO, trans::PolarFromCartesian) = print(io, "PolarFromCartesian()")
Base.show(io::IO, trans::CartesianFromPolar) = print(io, "CartesianFromPolar()")

function transform(::PolarFromCartesian, x::Point{2})
    Polar(sqrt(x[1]*x[1] + x[2]*x[2]), atan2(x[2], x[1]))
end

function transform_deriv{T}(::PolarFromCartesian, x::Point{2,T})
    r = sqrt(x[1]*x[1] + x[2]*x[2])
    f = x[2] / x[1]
    c = one(T)/(x[1]*(one(T) + f*f))
    @fsa [ x[1]/r    x[2]/r
          -f*c       c     ]
end
transform_deriv_params(::PolarFromCartesian, x::Point{2}) = error("PolarFromCartesian has no parameters")

function transform(::CartesianFromPolar, x::Polar)
    Point(x.r * cos(x.θ), x.r * sin(x.θ))
end
function transform_deriv(::CartesianFromPolar, x::Polar)
    sθ = sin(x.θ)
    cθ = cos(x.θ)
    @fsa [cθ  -x.r*sθ
          sθ   x.r*cθ ]
end
transform_deriv_params(::CartesianFromPolar, x::Polar) = error("CartesianFromPolar has no parameters")

Base.inv(::PolarFromCartesian) = CartesianFromPolar()
Base.inv(::CartesianFromPolar) = PolarFromCartesian()

compose(::PolarFromCartesian, ::CartesianFromPolar) = IdentityTransformation{Polar}()
compose(::CartesianFromPolar, ::PolarFromCartesian) = IdentityTransformation{Point{2}}()

#############################
### 3D Coordinate Systems ###
#############################
immutable Spherical{T}
    r::T
    θ::T
    ϕ::T
end
Base.show(io::IO, x::Spherical) = print(io, "Spherical(r=$(x.r), θ=$(x.θ) rad, ϕ=$(x.ϕ) rad)")
Base.isapprox(p1::Spherical, p2::Spherical; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.ϕ, p2.ϕ; kwargs...)

immutable Cylindrical{T}
    r::T
    θ::T
    z::T
end
Base.show(io::IO, x::Cylindrical) = print(io, "Cylindrical(r=$(x.r), θ=$(x.θ) rad, z=$(x.z))")
Base.isapprox(p1::Cylindrical, p2::Cylindrical; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...) && isapprox(p1.z, p2.z; kwargs...)

immutable SphericalFromCartesian <: AbstractTransformation{Spherical, Point{3}}; end
immutable CartesianFromSpherical <: AbstractTransformation{Point{3}, Spherical}; end
immutable CylindricalFromCartesian <: AbstractTransformation{Cylindrical, Point{3}}; end
immutable CartesianFromCylindrical <: AbstractTransformation{Point{3}, Cylindrical}; end
immutable CylindricalFromSpherical <: AbstractTransformation{Cylindrical, Spherical}; end
immutable SphericalFromCylindrical <: AbstractTransformation{Spherical, Cylindrical}; end

Base.show(io::IO, trans::SphericalFromCartesian) = print(io, "SphericalFromCartesian()")
Base.show(io::IO, trans::CartesianFromSpherical) = print(io, "CartesianFromSpherical()")
Base.show(io::IO, trans::CylindricalFromCartesian) = print(io, "CylindricalFromCartesian()")
Base.show(io::IO, trans::CartesianFromCylindrical) = print(io, "CartesianFromCylindrical()")
Base.show(io::IO, trans::CylindricalFromSpherical) = print(io, "CylindricalFromSpherical()")
Base.show(io::IO, trans::SphericalFromCylindrical) = print(io, "SphericalFromCylindrical()")

# Cartesian <-> Spherical
function transform(::SphericalFromCartesian, x::Point{3})
    Spherical(sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]), atan2(x[2],x[1]), atan(x[3]/sqrt(x[1]*x[1] + x[2]*x[2])))
end
function transform_deriv{T}(::SphericalFromCartesian, x::Point{3,T})
    r = sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3])
    rxy = sqrt(x[1]*x[1] + x[2]*x[2])
    fxy = x[2] / x[1]
    cxy = one(T)/(x[1]*(one(T) + fxy*fxy))
    f = -x[3]/(rxy*r*r)

    @fsa [ x[1]/r   x[2]/r  x[3]/r
          -fxy*cxy  cxy     zero(T)
           f*x[1]   f*x[2]  rxy/(r*r) ]
end
transform_deriv_params(::SphericalFromCartesian, x::Point{3}) = error("SphericalFromCartesian has no parameters")

function transform(::CartesianFromSpherical, x::Spherical)
    Point(x.r * cos(x.θ) * cos(x.ϕ), x.r * sin(x.θ) * cos(x.ϕ), x.r * sin(x.ϕ))
end
function transform_deriv{T}(::CartesianFromSpherical, x::Spherical{T})
    sθ = sin(x.θ)
    cθ = cos(x.θ)
    sϕ = sin(x.ϕ)
    cϕ = cos(x.ϕ)

    @fsa [cθ*cϕ -x.r*sθ*cϕ -x.r*cθ*sϕ
          sθ*cϕ  x.r*cθ*cϕ -x.r*sθ*sϕ
          sϕ     zero(T)    x.r * cϕ ]
end
transform_deriv_params(::CartesianFromSpherical, x::Spherical) = error("CartesianFromSpherical has no parameters")

# Cartesian <-> Cylindrical
function transform(::CylindricalFromCartesian, x::Point{3})
    Cylindrical(sqrt(x[1]*x[1] + x[2]*x[2]), atan2(x[2],x[1]), x[3])
end

function transform_deriv{T}(::CylindricalFromCartesian, x::Point{3,T})
    r = sqrt(x[1]*x[1] + x[2]*x[2])
    f = x[2] / x[1]
    c = one(T)/(x[1]*(one(T) + f*f))
    @fsa [x[1]/r   x[2]/r   zero(T)
          -f*c     c     zero(T)
          zero(T)  zero(T)  one(T) ]
end
transform_deriv_params(::CylindricalFromCartesian, x::Point{3}) = error("CylindricalFromCartesian has no parameters")

function transform(::CartesianFromCylindrical, x::Cylindrical)
    Point(x.r * cos(x.θ), x.r * sin(x.θ), x.z)
end
function transform_deriv{T}(::CartesianFromCylindrical, x::Cylindrical{T})
    sθ = sin(x.θ)
    cθ = cos(x.θ)
    @fsa [cθ      -x.r*sθ zero(T)
          sθ       x.r*cθ zero(T)
          zero(T) zero(T) one(T) ]
end
transform_deriv_params(::CartesianFromPolar, x::Cylindrical) = error("CartesianFromCylindrical has no parameters")

# Spherical <-> Cylindrical (TODO direct would be faster)
function transform(::CylindricalFromSpherical, x::Spherical)
    transform(CylindricalFromCartesian() , transform(CartesianFromSpherical(), x))
end
function transform_deriv(::CylindricalFromSpherical, x::Spherical)
    M1 = transform_deriv(CylindricalFromCartesian(), transform(CartesianFromSpherical(), x))
    M2 = transform_deriv(CartesianFromSpherical(), x)
    return M1*M2
end
transform_deriv_params(::CylindricalFromSpherical, x::Spherical) = error("CylindricalFromSpherical has no parameters")

function transform(::SphericalFromCylindrical, x::Cylindrical)
    transform(SphericalFromCartesian() , transform(CartesianFromCylindrical(), x))
end
function transform_deriv(::SphericalFromCylindrical, x::Cylindrical)
    M1 = transform_deriv(SphericalFromCartesian(), transform(CartesianFromCylindrical(), x))
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
compose(::SphericalFromCartesian,   ::CartesianFromSpherical)   = IdentityTransformation{Spherical}()
compose(::CartesianFromSpherical,   ::SphericalFromCartesian)   = IdentityTransformation{Point{3}}()
compose(::CylindricalFromCartesian, ::CartesianFromCylindrical) = IdentityTransformation{Cylindrical}()
compose(::CartesianFromCylindrical, ::CylindricalFromCartesian) = IdentityTransformation{Point{3}}()
compose(::CylindricalFromSpherical, ::SphericalFromCylindrical) = IdentityTransformation{Cylindrical}()
compose(::SphericalFromCylindrical, ::CylindricalFromSpherical) = IdentityTransformation{Spherical}()

# Cyclic compositions
compose(::SphericalFromCartesian,   ::CartesianFromCylindrical) = SphericalFromCylindrical()
compose(::CartesianFromSpherical,   ::SphericalFromCylindrical) = CartesianFromCylindrical()
compose(::CylindricalFromCartesian, ::CartesianFromSpherical)   = CylindricalFromSpherical()
compose(::CartesianFromCylindrical, ::CylindricalFromSpherical) = CartesianFromSpherical()
compose(::CylindricalFromSpherical, ::SphericalFromCartesian)   = CylindricalFromCartesian()
compose(::SphericalFromCylindrical, ::CylindricalFromCartesian) = SphericalFromCartesian()
