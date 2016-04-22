@testset "Core definitions" begin
    # Can't do much without a transformation, and the only one defined in
    # core.jl is IdentityTransformation
    T = Point{3,Float64}
    identity_trans = IdentityTransformation{T}()

    @test CoordinateTransformations.outtype(identity_trans) == T
    @test CoordinateTransformations.intype(identity_trans) == T

    @test inv(identity_trans) == identity_trans

    @test identity_trans ∘ identity_trans == identity_trans

    x = Point(1.0, 2.0, 3.0)
    @test transform(identity_trans, x) == x
end
