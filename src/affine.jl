abstract AbstractAffineMap <: Transformation

"""
    Translation(v) <: AbstractAffineMap
    Translation(dx, dy)       (2D)
    Translation(dx, dy, dz)   (3D)

Construct the `Translation` transformation for translating Cartesian points by
an offset `v = (dx, dy, ...)`
"""
immutable Translation{V <: AbstractVector} <: AbstractAffineMap
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
    LinearMap <: AbstractAffineMap
    LinearMap(M)

A general linear transformation, constructed using `LinearMap(M)`
for any `AbstractMatrix` `M`.
"""
immutable LinearMap{M <: AbstractMatrix} <: AbstractAffineMap
    m::M
end
Base.show(io::IO, trans::LinearMap)   = print(io, "LinearMap($(trans.M))") # TODO make this output more petite

function (trans::LinearMap{M}){M}(x)
    trans.m * x
end

Base.inv(trans::LinearMap) = LinearMap(inv(trans.m))

compose(t1::LinearMap, t2::LinearMap) = LinearMap(t1.m * t2.m)

function Base.isapprox(t1::LinearMap, t2::LinearMap; kwargs...)
    isapprox(t1.m, t2.m; kwargs...)
end

function Base.isapprox(t1::LinearMap, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.m), 0; kwargs...) &&
        isapprox(vecnorm(t2.v),0; kwargs...)
end

function Base.isapprox(t1::Translation, t2::LinearMap; kwargs...)
    isapprox(vecnorm(t1.v), 0; kwargs...) &&
        isapprox(vecnorm(t2.m),0; kwargs...)
end

function Base.:(==)(t1::LinearMap, t2::Translation)
    vecnorm(t1.m) == 0 &&
        0 == vecnorm(t2.v)
end

function Base.:(==)(t1::Translation, t2::LinearMap)
    vecnorm(t1.v) == 0 &&
        vecnorm(t2.m) == 0
end

transform_deriv(trans::LinearMap, x) = trans.m
# TODO transform_deriv_params

"""
    AffineMap <: AbstractAffineMap

A concrete affine transformation.  To construct the mapping `x -> M*x + v`, use

    AffineMap(M, v)

where `M` is a matrix and `v` a vector.  An arbitrary `Transformation` may be
converted into an affine approximation by linearizing about a point `x` using

    AffineMap(trans, [x])

For transformations which are already affine, `x` may be omitted.
"""
immutable AffineMap{M <: AbstractMatrix, V <: AbstractVector} <: AbstractAffineMap
    m::M
    v::V
end

function (trans::AffineMap{M,V}){M,V}(x)
    trans.m * x + trans.v
end

# Note: the expression `Tx - dT*Tx` will have large cancellation error for
# large Tx!  However, changing the order of applying the matrix and
# translation won't fix things, because then we'd have `Tx*(x-x0)` which
# also can incur large cancellation error in `x-x0`.
"""
    AffineMap(trans::Transformation, x0)

Create an Affine transformation corresponding to the differential transformation
of `x0 + dx` according to `trans`, i.e. the Affine transformation that is
locally most accurate in the vicinity of `x0`.
"""
function AffineMap(trans::Transformation, x0)
    dT = transform_deriv(trans, x0)
    Tx = trans(x0)
    AffineMap(dT, Tx - dT*x0)
end

Base.show(io::IO, trans::AffineMap) = print(io, "AffineMap($(trans.M), $(trans.v))") # TODO make this output more petite

function compose(t1::Translation, t2::LinearMap)
    AffineMap(t2.m, t1.v)
end

function compose(t1::LinearMap, t2::Translation)
    AffineMap(t1.m, t1.m * t2.v)
end

function compose(t1::AffineMap, t2::AffineMap)
    AffineMap(t1.m * t2.m, t1.v + t1.m * t2.v)
end

function compose(t1::AffineMap, t2::LinearMap)
    AffineMap(t1.m * t2.m, t1.v)
end

function compose(t1::LinearMap, t2::AffineMap)
    AffineMap(t1.m * t2.m, t1.m * t2.v)
end

function compose(t1::AffineMap, t2::Translation)
    AffineMap(t1.m, t1.v + t1.m * t2.v)
end

function compose(t1::Translation, t2::AffineMap)
    AffineMap(t2.m, t1.v + t2.v)
end

function Base.inv(trans::AffineMap)
    m_inv = inv(trans.m)
    AffineMap(m_inv, m_inv * (-trans.v))
end

function Base.isapprox(t1::AffineMap, t2::AffineMap; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::AffineMap, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.m), 0; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::Translation, t2::AffineMap; kwargs...)
    isapprox(vecnorm(t2.m), 0; kwargs...) &&
        isapprox(t1.v, t2.v; kwargs...)
end

function Base.isapprox(t1::AffineMap, t2::LinearMap; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(vecnorm(t1.v), 0; kwargs...)
end

function Base.isapprox(t1::LinearMap, t2::AffineMap; kwargs...)
    isapprox(t1.m, t2.m; kwargs...) &&
        isapprox(0, vecnorm(t2.v); kwargs...)
end


function Base.:(==)(t1::AffineMap, t2::Translation)
    vecnorm(t1.m) == 0 &&
        t1.v == t2.v
end

function Base.:(==)(t1::Translation, t2::AffineMap)
    vecnorm(t2.m) == 0 &&
        t1.v == t2.v
end

function Base.:(==)(t1::AffineMap, t2::LinearMap)
    t1.m == t2.m &&
        vecnorm(t1.v) == 0
end

function Base.:(==)(t1::LinearMap, t2::AffineMap)
    t1.m == t2.m &&
        0 == vecnorm(t2.v)
end

transform_deriv(trans::AffineMap, x) = trans.m
# TODO transform_deriv_params
