#####################################
### Interface for transformations ###
#####################################

# The external interface consists of transform(), Base.inv(),  compose() (or ∘),
# transform_deriv() and transform_deriv_params()

"""
`AbstractTransformation{OutType, InType}` defines a simple interface for
performing transformations. Subtypes should be able to apply a coordinate system
transformation on the correct data types by overloading `transform()`, and
usually would have the corresponding inverse transformation defined by `Base.inv()`.
Efficient compositions can optionally be defined by `compose()` (equivalently `∘`).
"""
abstract AbstractTransformation{OutType, InType}

# Some basic introspection
@inline intype{OutType, InType}(::Union{AbstractTransformation{OutType, InType}, Type{AbstractTransformation{OutType, InType}}}) = InType
@inline outtype{OutType, InType}(::Union{AbstractTransformation{OutType, InType}, Type{AbstractTransformation{OutType, InType}}}) = OutType

# Built-in identity transform
immutable IdentityTransformation{T} <: AbstractTransformation{T,T}; end

"""
    transform(trans::AbstractTransformation, x)

A transformation `trans` is explicitly applied to data x, returning the
coordinates in the new coordinate system.
"""
function transform{OutType, InType}(trans::AbstractTransformation{OutType, InType}, x)
    if isa(x, Vector) && eltype(x) <: InType
        [transform(trans, point) for point in x]
    else
        error("The transform of datatype $(typeof(x)) is not defined for transformation $trans.")
    end
end

transform{T}(::IdentityTransformation{T}, x::T) = x

"""
A `ComposedTransformation` simply executes two transformations successively, and
is the fallback output type of `compose()`.
"""
immutable ComposedTransformation{OutType, InType, T1 <: AbstractTransformation, T2 <: AbstractTransformation} <: AbstractTransformation{OutType, InType}
    t1::T1
    t2::T2

    function ComposedTransformation(trans1::T1, trans2::T2)
        check_composable(OutType, InType, trans1, trans2)
        new(trans1,trans2)
    end
end
function check_composable{OutType, InType}(out_type::Type{OutType},in_type::Type{InType},trans1::AbstractTransformation, trans2::AbstractTransformation)
    if in_type != intype(trans2)
        error("Can't compose transformations: input coordinates types $in_type and $(intype(trans2)) do not match.")
    elseif out_type != outtype(trans1)
        error("Can't compose transformations: output coordinates types $out_type and $(outtype(trans1)) do not match.")
    elseif intype(trans1) != outtype(trans2)
        error("Can't compose transformations: intermediate coordinates types $(intype(trans1)) and $(outtype(trans2)) do not match.")
    else
        # Shouldn't occur
        error("Unknown error: can't compose transformations $trans1 with $trans2 with input $in_type and output $out_type.")
    end
end
check_composable{OutType, InType, T}(::Type{OutType}, ::Type{InType}, ::AbstractTransformation{OutType,T}, ::AbstractTransformation{T,InType}) = nothing # Generates no code if they match... empty function

Base.show(io::IO, trans::ComposedTransformation) = print(io, "($(trans.t1) ∘ $(trans.t2))")

"""
    compose(trans1, trans2)
    trans1 ∘ trans2

Take two transformations and create a new transformation that is equivalent to
successively applying `trans2` to the coordinate, and then `trans1`. By default
will create a `ComposedTransformation`, however this method can be overloaded
for efficiency (e.g. two affine transformations naturally compose to a single
affine transformation).
"""
function compose(trans1::AbstractTransformation, trans2::AbstractTransformation)
    ComposedTransformation{outtype(trans1), intype(trans2), typeof(trans1), typeof(trans2)}(trans1, trans2)
end

compose{InOutType}(trans::IdentityTransformation{InOutType}, ::IdentityTransformation{InOutType}) = trans
compose{OutType, InType}(::IdentityTransformation{OutType}, trans::AbstractTransformation{OutType, InType}) = trans
compose{OutType, InType}(trans::AbstractTransformation{OutType, InType}, ::IdentityTransformation{InType}) = trans

const ∘ = compose

"""
    inv(trans::AbstractTransformation)

Returns the inverse (or reverse) of the transformation `trans`
"""
function Base.inv(trans::AbstractTransformation)
    error("Inverse transformation for $(typeof(trans)) has not been defined.")
end

Base.inv(trans::ComposedTransformation) = inv(trans.t2) ∘ inv(trans.t1)
Base.inv(trans::IdentityTransformation) = trans

"""
    transform_deriv(trans::AbstractTransformation, x)

A matrix describing how differentials on the parameters of `x` flow through to
the output of transformation `trans`.
"""
transform_deriv{T}(::IdentityTransformation{T}, x::T) = I

transform_deriv(::AbstractTransformation, x) = error("Differential matrix of transform $trans with input $x not defined")

"""
    transform_deriv_params(trans::AbstractTransformation, x)

A matrix describing how differentials on the parameters of `trans` flow through
to the output of transformation `trans` given input `x`.
"""
transform_deriv_params{T}(::IdentityTransformation{T}, x::T) = error("IdentityTransformation has no parameters")

transform_deriv(::AbstractTransformation, x) = error("Differential matrix of parameters of transform $trans with input $x not defined")
