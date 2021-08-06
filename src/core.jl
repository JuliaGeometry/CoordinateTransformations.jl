#####################################
### Interface for transformations ###
#####################################

# The external interface consists of transform(), Base.inv(),  compose() (or ∘),
# transform_deriv() and transform_deriv_params()

"""
The `Transformation` supertype defines a simple interface for performing
transformations. Subtypes should be able to apply a coordinate system
transformation on the correct data types by overloading the call method, and
usually would have the corresponding inverse transformation defined by `Base.inv()`.
Efficient compositions can optionally be defined by `compose()` (equivalently `∘`).
"""
abstract type Transformation end

"""
The `IdentityTransformation` is a singleton `Transformation` that returns the
input unchanged, similar to `identity`.
"""
struct IdentityTransformation <: Transformation; end

@inline (::IdentityTransformation)(x) = x

"""
A `ComposedTransformation` simply executes two transformations successively, and
is the fallback output type of `compose()`.
"""
struct ComposedTransformation{T1 <: Transformation, T2 <: Transformation} <: Transformation
    t1::T1
    t2::T2
end

Base.show(io::IO, trans::ComposedTransformation) = print(io, "($(trans.t1) ∘ $(trans.t2))")

@inline function (trans::ComposedTransformation)(x)
    trans.t1(trans.t2(x))
end

function Base.:(==)(trans1::ComposedTransformation, trans2::ComposedTransformation)
    (trans1.t1 == trans2.t1) && (trans1.t2 == trans2.t2)
end

function Base.isapprox(trans1::ComposedTransformation, trans2::ComposedTransformation; kwargs...)
    isapprox(trans1.t1, trans2.t1; kwargs...) && isapprox(trans1.t2, trans2.t2; kwargs...)
end

const compose = ∘

"""
    compose(trans1, trans2)
    trans1 ∘ trans2

Take two transformations and create a new transformation that is equivalent to
successively applying `trans2` to the coordinate, and then `trans1`. By default
will create a `ComposedTransformation`, however this method can be overloaded
for efficiency (e.g. two affine transformations naturally compose to a single
affine transformation).
"""
function compose(trans1::Transformation, trans2::Transformation)
    ComposedTransformation(trans1, trans2)
end

compose(trans::IdentityTransformation, ::IdentityTransformation) = trans
compose(::IdentityTransformation, trans::Transformation) = trans
compose(trans::Transformation, ::IdentityTransformation) = trans


"""
    inv(trans::Transformation)

Returns the inverse (or reverse) of the transformation `trans`
"""
function Base.inv(trans::Transformation)
    error("Inverse transformation for $(typeof(trans)) has not been defined.")
end

Base.inv(trans::ComposedTransformation) = inv(trans.t2) ∘ inv(trans.t1)
Base.inv(trans::IdentityTransformation) = trans

"""
    recenter(trans::Union{AbstractMatrix,Transformation}, origin::Union{AbstractVector, Tuple}) -> ctrans

Return a new transformation `ctrans` such that point `origin` serves
as the origin-of-coordinates for `trans`. Translation by `±origin`
occurs both before and after applying `trans`, so that if `trans` is
linear we have

    ctrans(origin) == origin

As a consequence, `recenter` only makes sense if the output space of
`trans` is isomorphic with the input space.

For example, if `trans` is a rotation matrix, then `ctrans` rotates
space around `origin`.
"""
function recenter(trans::Transformation, origin::AbstractVector)
    Translation(origin) ∘ trans ∘ Translation(-origin)
end
recenter(trans::Transformation, origin::Tuple) = recenter(trans, SVector(origin))


"""
    transform_deriv(trans::Transformation, x)

A matrix describing how differentials on the parameters of `x` flow through to
the output of transformation `trans`.
"""
transform_deriv(trans::Transformation, x) = error("Differential matrix of transform $trans with input $x not defined")

transform_deriv(::IdentityTransformation, x) = I

function transform_deriv(trans::ComposedTransformation, x)
    x2 = trans.t2(x)
    m1 = transform_deriv(trans.t1, x2)
    m2 = transform_deriv(trans.t2, x)
    return m1 * m2
end


"""
    transform_deriv_params(trans::AbstractTransformation, x)

A matrix describing how differentials on the parameters of `trans` flow through
to the output of transformation `trans` given input `x`.
"""
transform_deriv_params(trans::Transformation, x) = error("Differential matrix of parameters of transform $trans with input $x not defined")

transform_deriv_params(::IdentityTransformation, x) = error("IdentityTransformation has no parameters")

function transform_deriv_params(trans::ComposedTransformation, x)
    x2 = trans.t2(x)
    m1 = transform_deriv(trans.t1, x2)
    p2 = transform_deriv_params(trans.t2, x)
    p1 = transform_deriv_params(trans.t1, x2)
    return hcat(p1, m1*p2)
end
