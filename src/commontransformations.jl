# Some common transformations are defined here

###################
### Translation ###
###################

immutable Translation{N,T} <: AbstractTransformation{Point{N,T}, Point{N,T}}
    dx::Point{N,T}
end

Base.show(io::IO, trans::Translation) = print(io, "Translation($(trans.dx))")

function transform{N,T}(trans::Translation{N,T}, x::Point{N,T})
    x + trans.dx
end

Base.inv(trans::Translation) = Translation(-trans.dx)

function compose{N,T}(trans1::Translation{N,T}, trans2::Translation{N,T})
    Translation(trans1.dx + trans2.dx)
end
