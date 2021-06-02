abstract type AbstractAffineMap <: Transformation end

"""
    Translation(v) <: AbstractAffineMap
    Translation(dx, dy)       (2D)
    Translation(dx, dy, dz)   (3D)

Construct the `Translation` transformation for translating Cartesian points by
an offset `v = (dx, dy, ...)`
"""
struct Translation{V} <: AbstractAffineMap
    translation::V
end
Translation(x::Tuple) = Translation(SVector(x))
Translation(x,y) = Translation(SVector(x,y))
Translation(x,y,z) = Translation(SVector(x,y,z))
Base.show(io::IO, trans::Translation) = print(io, "Translation$((trans.translation...,))")

function (trans::Translation{V})(x) where {V}
    x + trans.translation
end

Base.inv(trans::Translation) = Translation(-trans.translation)

function compose(trans1::Translation, trans2::Translation)
    Translation(trans1.translation + trans2.translation)
end

transform_deriv(trans::Translation, x) = I
transform_deriv_params(trans::Translation, x) = I

function Base.isapprox(t1::Translation, t2::Translation; kwargs...)
    isapprox(t1.translation, t2.translation; kwargs...)
end


"""
    LinearMap <: AbstractAffineMap
    LinearMap(M)

A general linear transformation, constructed using `LinearMap(M)`
for any matrix-like object `M`.
"""
struct LinearMap{M} <: AbstractAffineMap
    linear::M
end
Base.show(io::IO, trans::LinearMap) = print(io, "LinearMap($(trans.linear))") # TODO make this output more petite

function (trans::LinearMap{M})(x) where {M}
    trans.linear * x
end
(trans::LinearMap{M})(x::Tuple) where {M} = trans(SVector(x))

Base.inv(trans::LinearMap) = LinearMap(inv(trans.linear))

compose(t1::LinearMap, t2::LinearMap) = LinearMap(t1.linear * t2.linear)

function Base.isapprox(t1::LinearMap, t2::LinearMap; kwargs...)
    isapprox(t1.linear, t2.linear; kwargs...)
end

function Base.isapprox(t1::LinearMap, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.linear), 0; kwargs...) &&
        isapprox(vecnorm(t2.translation),0; kwargs...)
end

function Base.isapprox(t1::Translation, t2::LinearMap; kwargs...)
    isapprox(vecnorm(t1.translation), 0; kwargs...) &&
        isapprox(vecnorm(t2.linear),0; kwargs...)
end

function Base.:(==)(t1::LinearMap, t2::Translation)
    vecnorm(t1.linear) == 0 &&
        0 == vecnorm(t2.translation)
end

function Base.:(==)(t1::Translation, t2::LinearMap)
    vecnorm(t1.translation) == 0 &&
        vecnorm(t2.linear) == 0
end

transform_deriv(trans::LinearMap, x) = trans.linear
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
struct AffineMap{M, V} <: AbstractAffineMap
    linear::M
    translation::V
end

function (trans::AffineMap)(x)
    l = LinearMap(trans.linear)
    t = Translation(trans.translation)
    t(l(x))
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

Base.show(io::IO, trans::AffineMap) = print(io, "AffineMap($(trans.linear), $(trans.translation))") # TODO make this output more petite

function compose(t1::Translation, t2::LinearMap)
    AffineMap(t2.linear, t1.translation)
end

function compose(t1::LinearMap, t2::Translation)
    AffineMap(t1.linear, t1.linear * t2.translation)
end

function compose(t1::AffineMap, t2::AffineMap)
    AffineMap(t1.linear * t2.linear, t1.translation + t1.linear * t2.translation)
end

function compose(t1::AffineMap, t2::LinearMap)
    AffineMap(t1.linear * t2.linear, t1.translation)
end

function compose(t1::LinearMap, t2::AffineMap)
    AffineMap(t1.linear * t2.linear, t1.linear * t2.translation)
end

function compose(t1::AffineMap, t2::Translation)
    AffineMap(t1.linear, t1.translation + t1.linear * t2.translation)
end

function compose(t1::Translation, t2::AffineMap)
    AffineMap(t2.linear, t1.translation + t2.translation)
end

function Base.inv(trans::AffineMap)
    m_inv = inv(trans.linear)
    AffineMap(m_inv, m_inv * (-trans.translation))
end

function Base.isapprox(t1::AffineMap, t2::AffineMap; kwargs...)
    isapprox(t1.linear, t2.linear; kwargs...) &&
        isapprox(t1.translation, t2.translation; kwargs...)
end

function Base.isapprox(t1::AffineMap, t2::Translation; kwargs...)
    isapprox(vecnorm(t1.linear), 0; kwargs...) &&
        isapprox(t1.translation, t2.translation; kwargs...)
end

function Base.isapprox(t1::Translation, t2::AffineMap; kwargs...)
    isapprox(vecnorm(t2.linear), 0; kwargs...) &&
        isapprox(t1.translation, t2.translation; kwargs...)
end

function Base.isapprox(t1::AffineMap, t2::LinearMap; kwargs...)
    isapprox(t1.linear, t2.linear; kwargs...) &&
        isapprox(vecnorm(t1.translation), 0; kwargs...)
end

function Base.isapprox(t1::LinearMap, t2::AffineMap; kwargs...)
    isapprox(t1.linear, t2.linear; kwargs...) &&
        isapprox(0, vecnorm(t2.translation); kwargs...)
end


function Base.:(==)(t1::AffineMap, t2::Translation)
    vecnorm(t1.linear) == 0 &&
        t1.translation == t2.translation
end

function Base.:(==)(t1::Translation, t2::AffineMap)
    vecnorm(t2.linear) == 0 &&
        t1.translation == t2.translation
end

function Base.:(==)(t1::AffineMap, t2::LinearMap)
    t1.linear == t2.linear &&
        vecnorm(t1.translation) == 0
end

function Base.:(==)(t1::LinearMap, t2::AffineMap)
    t1.linear == t2.linear &&
        0 == vecnorm(t2.translation)
end

recenter(trans::AbstractMatrix, origin::Union{AbstractVector, Tuple}) = recenter(LinearMap(trans), origin)

transform_deriv(trans::AffineMap, x) = trans.linear
# TODO transform_deriv_params
