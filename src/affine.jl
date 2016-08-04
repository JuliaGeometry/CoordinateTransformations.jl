abstract AbstractAffineTransformation <: Transformation

"""
    Translation(v) <: AbstractAffineTransformation
    Translation(dx, dy)       (2D)
    Translation(dx, dy, dz)   (3D)

Construct the `Translation` transformation for translating Cartesian points by
an offset `v = (dx, dy, ...)`
"""
immutable Translation{V <: AbstractVector} <: AbstractAffineTransformation
    v::V
end
Translation(x::Tuple) = Translation(SVector(x))
Translation(x,y) = Translation(SVector(x,y))
Translation(x,y,z) = Translation(SVector(x,y,z))
Base.show(io::IO, trans::Translation) = print(io, "Translation$((trans.dx...))")

function (trans::Translation{V}){V}(x)
    x + trans.v
end

Base.inv(trans::Translation) = Translation(-trans.v)

function compose(trans1::Translation, trans2::Translation)
    Translation(trans1.v + trans2.v)
end

transform_deriv(trans::Translation, x) = I
transform_deriv_params(trans::Translation, x) = I

function Base.isapprox(t1::Translation, t2::Translation; kwargs...)
    isapprox(t1.v, t2.v; kwargs...)
end


"""
    LinearTransformation <: AbstractAffineTransformation
    LinearTransformation(M)

A general linear transformation, constructed using `LinearTransformation(M)`
for any `AbstractMatrix` `M`.
"""
immutable LinearTransformation{M <: AbstractMatrix} <: AbstractAffineTransformation
    m::M
end
Base.show(io::IO, trans::LinearTransformation)   = print(io, "LinearTransformation($(trans.M))") # TODO make this output more petite

function (trans::LinearTransformation{M}){M}(x)
    trans.m * x
end

Base.inv(trans::LinearTransformation) = LinearTransformation(inv(trans.m))

compose(t1::LinearTransformation, t2::LinearTransformation) = LinearTransformation(t1.m * t2.m)

function Base.isapprox(t1::LinearTransformation, t2::LinearTransformation; kwargs...)
    isapprox(t1.m, t2.m; kwargs...)
end

function Base.isapprox(t1::LinearTransformation, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.m), 0; kwargs...) &&
        isapprox(vecnorm(t2.v),0; kwargs...)
end

function Base.isapprox(t1::Translation, t2::LinearTransformation; kwargs...)
    isapprox(vecnorm(t1.v), 0; kwargs...) &&
        isapprox(vecnorm(t2.m),0; kwargs...)
end

function Base.:(==)(t1::LinearTransformation, t2::Translation)
    vecnorm(t1.m) == 0 &&
        0 == vecnorm(t2.v)
end

function Base.:(==)(t1::Translation, t2::LinearTransformation)
    vecnorm(t1.v) == 0 &&
        vecnorm(t2.m) == 0
end

transform_deriv(trans::LinearTransformation, x) = trans.m
# TODO transform_deriv_params

"""
    AffineTransformation <: AbstractAffineTransformation

A concrete affine transformation.  To construct the mapping `x -> M*x + v`, use

    AffineTransformation(M, v)

where `M` is a matrix and `v` a vector.  An arbitrary `Transformation` may be
converted into an affine approximation by linearizing about a point `x` using

    AffineTransformation(trans, [x])

For transformations which are already affine, `x` may be omitted.
"""
immutable AffineTransformation{M <: AbstractMatrix, V <: AbstractVector} <: AbstractAffineTransformation
    m::M
    v::V
end

function (trans::AffineTransformation{M,V}){M,V}(x)
    trans.m * x + trans.v
end

# Note: the expression `Tx - dT*Tx` will have large cancellation error for
# large Tx!  However, changing the order of applying the matrix and
# translation won't fix things, because then we'd have `Tx*(x-x0)` which
# also can incur large cancellation error in `x-x0`.
"""
    AffineTransformation(trans::Transformation, x0)

Create an Affine transformation corresponding to the differential transformation
of `x0 + dx` according to `trans`, i.e. the Affine transformation that is
locally most accurate in the vicinity of `x0`.
"""
function AffineTransformation(trans::Transformation, x0)
    dT = transform_deriv(trans, x0)
    Tx = trans(x0)
    AffineTransformation(dT, Tx - dT*x0)
end

Base.show(io::IO, trans::AffineTransformation) = print(io, "AffineTransformation($(trans.M), $(trans.v))") # TODO make this output more petite

function compose(t1::Translation, t2::LinearTransformation)
    AffineTransformation(t2.m, t1.v)
end

function compose(t1::LinearTransformation, t2::Translation)
    AffineTransformation(t1.m, t1.m * t2.v)
end

function compose(t1::AffineTransformation, t2::AffineTransformation)
    AffineTransformation(t1.m * t2.m, t1.v + t1.m * t2.v)
end

function compose(t1::AffineTransformation, t2::LinearTransformation)
    AffineTransformation(t1.m * t2.m, t1.v)
end

function compose(t1::LinearTransformation, t2::AffineTransformation)
    AffineTransformation(t1.m * t2.m, t1.m * t2.v)
end

function compose(t1::AffineTransformation, t2::Translation)
    AffineTransformation(t1.m, t1.v + t1.m * t2.v)
end

function compose(t1::Translation, t2::AffineTransformation)
    AffineTransformation(t2.m, t1.v + t2.v)
end

function Base.inv(trans::AffineTransformation)
    m_inv = inv(trans.m)
    AffineTransformation(m_inv, m_inv * (-trans.v))
end

function Base.isapprox(t1::AffineTransformation, t2::AffineTransformation; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::AffineTransformation, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.m), 0; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::Translation, t2::AffineTransformation; kwargs...)
    isapprox(vecnorm(t2.m), 0; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::AffineTransformation, t2::LinearTransformation; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(vecnorm(t1.v), 0; kwargs...)
end

function Base.isapprox(t1::LinearTransformation, t2::AffineTransformation; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(0, vecnorm(t2.v); kwargs...)
end


function Base.:(==)(t1::AffineTransformation, t2::Translation)
    vecnorm(t1.m) == 0 &&
        t1.v == t2.v
end

function Base.:(==)(t1::Translation, t2::AffineTransformation)
    vecnorm(t2.m) == 0 &&
        t1.v == t2.v
end

function Base.:(==)(t1::AffineTransformation, t2::LinearTransformation)
    t1.m == t2.m &&
        vecnorm(t1.v) == 0
end

function Base.:(==)(t1::LinearTransformation, t2::AffineTransformation)
    t1.m == t2.m &&
        0 == vecnorm(t2.v)
end

transform_deriv(trans::AffineTransformation, x) = trans.m
# TODO transform_deriv_params
