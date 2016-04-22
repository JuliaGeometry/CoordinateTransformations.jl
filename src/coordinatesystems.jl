#############################
### 2D Coordinate systems ###
#############################
immutable Polar{T}
    r::T
    θ::T
end
Base.show(io::IO, x::Polar) = print(io, "Polar(r=$(x.r), θ=$(x.θ) rad)")
Base.isapprox(p1::Polar, p2::Polar; kwargs...) = isapprox(p1.r, p2.r; kwargs...) && isapprox(p1.θ, p2.θ; kwargs...)

immutable PolarFromCartesian{T} <: AbstractTransformation{Polar{T}, Point{2,T}}; end
immutable CartesianFromPolar{T} <: AbstractTransformation{Point{2,T}, Polar{T}}; end

Base.show{T}(io::IO, trans::PolarFromCartesian{T}) = print(io, "PolarFromCartesian{$T}()")
Base.show{T}(io::IO, trans::CartesianFromPolar{T}) = print(io, "CartesianFromPolar{$T}()")

# TODO: Branch-cut is a 3π/2, could move it to ±π (also for Spherical and Cylindrical)
function transform{T}(::PolarFromCartesian{T}, x::Point{2,T})
    Polar(sqrt(x[1]*x[1] + x[2]*x[2]), atan(x[2]/x[1]) + pi*(x[1]<0))
end

function transform{T}(::CartesianFromPolar{T}, x::Polar{T})
    Point(x.r * cos(x.θ), x.r * sin(x.θ))
end

Base.inv{T}(::PolarFromCartesian{T}) = CartesianFromPolar{T}()
Base.inv{T}(::CartesianFromPolar{T}) = PolarFromCartesian{T}()

compose{T}(::PolarFromCartesian{T}, ::CartesianFromPolar{T}) = IdentityTransformation{Polar{T}}()
compose{T}(::CartesianFromPolar{T}, ::PolarFromCartesian{T}) = IdentityTransformation{Point{2,T}}()

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

immutable SphericalFromCartesian{T} <: AbstractTransformation{Spherical{T}, Point{3,T}}; end
immutable CartesianFromSpherical{T} <: AbstractTransformation{Point{3,T}, Spherical{T}}; end
immutable CylindricalFromCartesian{T} <: AbstractTransformation{Cylindrical{T}, Point{3,T}}; end
immutable CartesianFromCylindrical{T} <: AbstractTransformation{Point{3,T}, Cylindrical{T}}; end
immutable CylindricalFromSpherical{T} <: AbstractTransformation{Cylindrical{T}, Spherical{T}}; end
immutable SphericalFromCylindrical{T} <: AbstractTransformation{Spherical{T}, Cylindrical{T}}; end

Base.show{T}(io::IO, trans::SphericalFromCartesian{T}) = print(io, "SphericalFromCartesian{$T}()")
Base.show{T}(io::IO, trans::CartesianFromSpherical{T}) = print(io, "CartesianFromSpherical{$T}()")
Base.show{T}(io::IO, trans::CylindricalFromCartesian{T}) = print(io, "CylindricalFromCartesian{$T}()")
Base.show{T}(io::IO, trans::CartesianFromCylindrical{T}) = print(io, "CartesianFromCylindrical{$T}()")
Base.show{T}(io::IO, trans::CylindricalFromSpherical{T}) = print(io, "CylindricalFromSpherical{$T}()")
Base.show{T}(io::IO, trans::SphericalFromCylindrical{T}) = print(io, "SphericalFromCylindrical{$T}()")

# Cartesian <-> Spherical
function transform{T}(::SphericalFromCartesian{T}, x::Point{3,T})
    Spherical(sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]), atan(x[2]/x[1]) + pi*(x[1]<0), atan(x[3]/sqrt(x[1]*x[1] + x[2]*x[2])))
end

function transform{T}(::CartesianFromSpherical{T}, x::Spherical{T})
    Point(x.r * cos(x.θ) * cos(x.ϕ), x.r * sin(x.θ) * cos(x.ϕ), x.r * sin(x.ϕ))
end

# Cartesian <-> Cylindrical
function transform{T}(::CylindricalFromCartesian{T}, x::Point{3,T})
    Cylindrical(sqrt(x[1]*x[1] + x[2]*x[2]), atan(x[2]/x[1]) + pi*(x[1]<0), x[3])
end

function transform{T}(::CartesianFromCylindrical{T}, x::Cylindrical{T})
    Point(x.r * cos(x.θ), x.r * sin(x.θ), x.z)
end

# Spherical <-> Cylindrical (TODO direct could be faster)
function transform{T}(::CylindricalFromSpherical{T}, x::Spherical{T})
    transform(CylindricalFromCartesian{T}() , transform(CartesianFromSpherical{T}(), x))
end

function transform{T}(::SphericalFromCylindrical{T}, x::Cylindrical{T})
    transform(SphericalFromCartesian{T}() , transform(CartesianFromCylindrical{T}(), x))
end

Base.inv{T}(::SphericalFromCartesian{T}) = CartesianFromSpherical{T}()
Base.inv{T}(::CartesianFromSpherical{T}) = SphericalFromCartesian{T}()
Base.inv{T}(::CylindricalFromCartesian{T}) = CartesianFromCylindrical{T}()
Base.inv{T}(::CartesianFromCylindrical{T}) = CylindricalFromCartesian{T}()
Base.inv{T}(::CylindricalFromSpherical{T}) = SphericalFromCylindrical{T}()
Base.inv{T}(::SphericalFromCylindrical{T}) = CylindricalFromSpherical{T}()

# Inverse composition
compose{T}(::SphericalFromCartesian{T}, ::CartesianFromSpherical{T})     = IdentityTransformation{Spherical{T}}()
compose{T}(::CartesianFromSpherical{T}, ::SphericalFromCartesian{T})     = IdentityTransformation{Point{3,T}}()
compose{T}(::CylindricalFromCartesian{T}, ::CartesianFromCylindrical{T}) = IdentityTransformation{Cylindrical{T}}()
compose{T}(::CartesianFromCylindrical{T}, ::CylindricalFromCartesian{T}) = IdentityTransformation{Point{3,T}}()
compose{T}(::CylindricalFromSpherical{T}, ::SphericalFromCylindrical{T}) = IdentityTransformation{Cylindrical{T}}()
compose{T}(::SphericalFromCylindrical{T}, ::CylindricalFromSpherical{T}) = IdentityTransformation{Spherical{T}}()

# Cyclic compositions
compose{T}(::SphericalFromCartesian{T}, ::CartesianFromCylindrical{T})   = SphericalFromCylindrical{T}()
compose{T}(::CartesianFromSpherical{T}, ::SphericalFromCylindrical{T})   = CartesianFromCylindrical{T}()
compose{T}(::CylindricalFromCartesian{T}, ::CartesianFromSpherical{T})   = CylindricalFromSpherical{T}()
compose{T}(::CartesianFromCylindrical{T}, ::CylindricalFromSpherical{T}) = CartesianFromSpherical{T}()
compose{T}(::CylindricalFromSpherical{T}, ::SphericalFromCartesian{T})   = CylindricalFromCartesian{T}()
compose{T}(::SphericalFromCylindrical{T}, ::CylindricalFromCartesian{T}) = SphericalFromCartesian{T}()
