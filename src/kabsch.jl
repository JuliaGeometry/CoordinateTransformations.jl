# Compute rigid and similarity transformations between point sets

# For rigid transformations, we use:
# Kabsch, Wolfgang. "A discussion of the solution for the best rotation to
# relate two sets of vectors." Acta Crystallographica Section A: Crystal
# Physics, Diffraction, Theoretical and General Crystallography 34.5 (1978):
# 827-828.
# This has been generalized to support weighted points:
# https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
# We add the component for similarity transformations from:
# Umeyama, Shinji. "Least-squares estimation of transformation parameters
# between two point patterns." IEEE Transactions on Pattern Analysis & Machine
# Intelligence 13.04 (1991): 376-380.

# See also
# https://en.wikipedia.org/wiki/Kabsch_algorithm


# All matrices are DxN, where N is the number of positions and D is the dimensionality

# Here, P is the probe (to be rotated) and Q is the refereence

# `kabsch_centered` assumes P and Q are already centered at the origin
# returns the rotation (optionally with scaling) for alignment
function kabsch_centered(P, Q, w; scale::Bool=false, svd::F = LinearAlgebra.svd) where F
    @assert size(P) == size(Q)
    W = Diagonal(w/sum(w))
    H = P*W*Q'
    U,Σ,V = svd(H)
    Ddiag = ones(eltype(H), size(H,1))
    Ddiag[end] = sign(det(V*U'))
    c = scale ? sum(Σ .* Ddiag) / sum(P .* (P*W)) : 1
    return LinearMap(V * Diagonal(c * Ddiag) * U')
end

"""
    kabsch(from_points => to_points, w=ones(npoints); scale::Bool=false, svd=LinearAlgebra.svd) → trans

Compute the rigid transformation (or similarity transformation, if `scale=true`)
that aligns `from_points` to `to_points` in a least-squares sense.

Optionally specify the non-negative weights `w` for each point. The default value of the weight
is 1 for each point.

For
differentiability, use `svd = GenericLinearAlgebra.svd` or other differentiable
singular value decomposition.
"""
function kabsch(pr::Pair{<:AbstractMatrix, <:AbstractMatrix}, w::AbstractVector=ones(size(pr.first,2)); scale::Bool=false, kwargs...)
    P, Q = pr
    any(<(0), w) && throw(ArgumentError("weights must be non-negative"))
    all(iszero, w) && throw(ArgumentError("weights must not all be zero"))
    wn = w/sum(w)
    centerP, centerQ = P*wn, Q*wn
    R = kabsch_centered(P .- centerP, Q .- centerQ, w; scale, kwargs...)
    return inv(Translation(-centerQ)) ∘ R ∘ Translation(-centerP)
end
kabsch((from_points, to_points)::Pair, args...; kwargs...) = kabsch(column_matrix(from_points) => column_matrix(to_points), args...; kwargs...)
