@testset "Core definitions" begin
    # Can't do much without a transformation, and the only one defined in
    # core.jl is IdentityTransformation... will have to test
    # ComposeTransformation in test/commontransformations.jl
    T = Point{3}
    identity_trans = IdentityTransformation{T}()

    @test CoordinateTransformations.outtype(identity_trans) == T
    @test CoordinateTransformations.intype(identity_trans) == T
    @test CoordinateTransformations.outtype(IdentityTransformation{T}) == T
    @test CoordinateTransformations.intype(IdentityTransformation{T}) == T

    @test inv(identity_trans) == identity_trans

    @test identity_trans ∘ identity_trans == identity_trans

    x = Point(1.0, 2.0, 3.0)
    @test transform(identity_trans, x) == x

    v = [Point(1.0, 2.0, 3.0), Point(1.0, 2.0, 3.0)]
    @test transform(identity_trans, v) == v
end
