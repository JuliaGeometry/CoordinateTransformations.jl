@testset "Core definitions" begin
    # Can't do much without a transformation, and the only one defined in
    # core.jl is IdentityTransformation... will have to test
    # ComposeTransformation in test/commontransformations.jl
    identity_trans = IdentityTransformation()

    @test inv(identity_trans) == identity_trans

    @test identity_trans ∘ identity_trans == identity_trans

    x = SVector(1.0, 2.0, 3.0)
    @test identity_trans(x) == x
end
