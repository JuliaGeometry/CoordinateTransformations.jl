@testset "Common Transformations" begin
    @testset "Translation" begin
        x = Point(1.0, 2.0)
        trans = Translation(Point(2.0, -1.0))

        # Inverse
        @test inv(trans) == Translation(Point(-2.0, 1.0))

        # Composition
        @test trans ∘ trans == Translation(Point(4.0, -2.0))

        # Transform
        @test transform(trans, x) == Point(3.0, 1.0)
    end
end
