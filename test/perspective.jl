@testset "Perspective transformation" begin
    @test PerspectiveMap()([2.0, -1.0, 0.5]) ≈ [4.0, -2.0]

    if VERSION >= v"0.6.0-dev.948"
        @test cameramap(2.0)                  ≈ LinearMap(UniformScaling(2.0)) ∘ PerspectiveMap()
        @test cameramap((1.1,2.2))            ≈ LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
        @test cameramap(1.1,       (3.3,4.4)) ≈ Translation([-3.3,-4.4]) ∘ LinearMap(UniformScaling(1.1)) ∘ PerspectiveMap()
        @test cameramap((1.1,2.2), (3.3,4.4)) ≈ Translation([-3.3,-4.4]) ∘ LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
    else
        # `isapprox` for `UniformScaling` not defined. Just put a few points through and
        # test that we get the same results
        points = [[1.0,1.0,1.0], [-1.0,1.0,1.0], [-1.0,-1.0,1.0], [1.0,-1.0,1.0]]

        trans1 = cameramap(2.0)
        trans2 = LinearMap(UniformScaling(2.0)) ∘ PerspectiveMap()
        @test all(isapprox.(trans1.(points), trans2.(points)))

        trans1 = cameramap((1.1,2.2))
        trans2 = LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
        @test all(isapprox.(trans1.(points), trans2.(points)))

        trans1 = cameramap(1.1,       (3.3,4.4))
        trans2 = Translation([-3.3,-4.4]) ∘ LinearMap(UniformScaling(1.1)) ∘ PerspectiveMap()
        @test all(isapprox.(trans1.(points), trans2.(points)))

        trans1 = cameramap((1.1,2.2), (3.3,4.4))
        trans2 = Translation([-3.3,-4.4]) ∘ LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
        @test all(isapprox.(trans1.(points), trans2.(points)))

    end
end
