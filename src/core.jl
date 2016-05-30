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

Furthermore, transformations can be combined in a tuple to be applied in parallel
to a tuple of inputs (for example, performing a rotation on a spatial variable
while leaving the time variable constant). This becomes particularly powerful
when
"""
abstract AbstractTransformation{OutType, InType}

# Some basic introspection
@inline intype{OutType, InType}(::AbstractTransformation{OutType, InType}) = InType
@inline outtype{OutType, InType}(::AbstractTransformation{OutType, InType}) = OutType

# Unfortunately some operations on abstract types are difficult
@generated function intype{T <: AbstractTransformation}(::Type{T})
    S = T
    while S.name != AbstractTransformation.name
        S = super(S)
        if S == Any
            str = "Error determining intype of $T"
            return :(error(str))
        end
    end

    return :($(S.parameters[2]))
end

@generated function outtype{T <: AbstractTransformation}(::Type{T})
    S = T
    while S.name != AbstractTransformation.name
        S = super(S)
        if S == Any
            str = "Error determining outtype of $T"
            return :(error(str))
        end
    end

    return :($(S.parameters[1]))
end

# TODO generated functions for intype and outtype of tuples of AbstractTransformations


# Built-in identity transform
immutable IdentityTransformation{T} <: AbstractTransformation{T,T}; end

"""
    transform(trans::AbstractTransformation, x)

A transformation `trans` is explicitly applied to data x, returning the
coordinates in the new coordinate system.
"""
function transform{OutType, InType}(trans::AbstractTransformation{OutType, InType}, x)
    if isa(x, Vector) && eltype(x) <: InType # TODO remove this function for julia-0.5 where vectorized functions are falling out of favour
        [transform(trans, point) for point in x]
    else
        error("The transform of datatype $(typeof(x)) is not defined for transformation $trans.")
    end
end

"""
transform((trans1, trans2, ...), (x1, x2, ...))

A set of transformations `trans1`, etc, are applied in parallel to data `x1`, etc,
returning a tuple of the transformed coordinates.
"""
function transform{N}(trans::NTuple{N,AbstractTransformation}, x::NTuple{N})
    # TODO generated function for type stability in Julia 0.4
    return ntuple(i->transform(trans[i], x[i]), Val{N})
end

transform{T}(::IdentityTransformation{T}, x::T) = x


"""
A `ComposedTransformation` simply executes two transformations successively, and
is the fallback output type of `compose()`.
"""
immutable ComposedTransformation{OutType, InType, T1 <: Union{AbstractTransformation, Tuple{Vararg{AbstractTransformation}}}, T2 <: Union{AbstractTransformation, Tuple{Vararg{AbstractTransformation}}}} <: AbstractTransformation{OutType, InType}
    t1::T1
    t2::T2

    function ComposedTransformation(trans1::T1, trans2::T2)
        check_composable(OutType, InType, trans1, trans2)
        new(trans1,trans2)
    end
end
@generated function check_composable{OutType, InType}(out_type::Type{OutType},in_type::Type{InType},trans1::Union{AbstractTransformation, Tuple{Vararg{AbstractTransformation}}}, trans2::Union{AbstractTransformation, Tuple{Vararg{AbstractTransformation}}})
    if InType != intype(trans2)
        str = "Can't compose transformations: input coordinates types $InType and $(intype(trans2)) do not match."
        error(str)
    elseif OutType != outtype(trans1)
        str = "Can't compose transformations: output coordinates types $OutType and $(outtype(trans1)) do not match."
        error(str)
    elseif typeintersect(intype(trans1), outtype(trans2)) == Union{}
        error("Can't compose transformations: intermediate coordinates types $(intype(trans1)) and $(outtype(trans2)) do not intersect.")
    else
        return nothing
    end
end
check_composable{OutType, InType, T}(::Type{OutType}, ::Type{InType}, ::AbstractTransformation{OutType,T}, ::AbstractTransformation{T,InType}) = nothing # Generates no code if they match... empty function

Base.show(io::IO, trans::ComposedTransformation) = print(io, "($(trans.t1) ∘ $(trans.t2))")

@inline function transform{OutType, InType}(trans::ComposedTransformation{OutType, InType}, x::InType)
    transform(trans.t1, transform(trans.t2, x))
end

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

# TODO compose for tuples of AbstractTransformations (generated function)

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

# TODO compose for inverse of AbstractTransformations (generated function)

"""
    transform_deriv(trans::AbstractTransformation, x)

A matrix describing how differentials on the parameters of `x` flow through to
the output of transformation `trans`.
"""
transform_deriv(::AbstractTransformation, x) = error("Differential matrix of transform $trans with input $x not defined")

transform_deriv{T}(::IdentityTransformation{T}, x::T) = I

function transform_deriv{OutType, InType}(trans::ComposedTransformation{OutType, InType}, x::InType)
    x2 = transform(trans.t2, x)
    m1 = transform_deriv(trans.t1, x2)
    m2 = transform_deriv(trans.t2, x)
    return m1 * m2
end

# TODO compose for derivatives of AbstractTransformations (generated function)

"""
    transform_deriv_params(trans::AbstractTransformation, x)

A matrix describing how differentials on the parameters of `trans` flow through
to the output of transformation `trans` given input `x`.
"""
transform_deriv(::AbstractTransformation, x) = error("Differential matrix of parameters of transform $trans with input $x not defined")

transform_deriv_params{T}(::IdentityTransformation{T}, x::T) = error("IdentityTransformation has no parameters")

function transform_deriv_params{OutType, InType}(trans::ComposedTransformation{OutType, InType}, x::InType)
    x2 = transform(trans.t2, x)
    m1 = transform_deriv(trans.t1, x2)
    p2 = transform_deriv_params(trans.t2, x)
    p1 = transform_deriv_params(trans.t1, x2)
    return hcat(p1, m1*p2)
end

# TODO compose for parameter derivatives of AbstractTransformations (generated function)
