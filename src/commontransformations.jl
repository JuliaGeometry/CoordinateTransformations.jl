# Some common transformations are defined here

###################
### Translation ###
###################

immutable Translation{N,T} <: AbstractTransformation{Point{N}, Point{N}}
    dx::Point{N,T}
end
Base.show(io::IO, trans::Translation) = print(io, "Translation($(trans.dx))")

function transform{N}(trans::Translation{N}, x::Point{N})
    x + trans.dx
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
