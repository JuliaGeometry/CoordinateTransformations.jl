@testset "Perspective transformation" begin
    @test PerspectiveMap()([2.0, -1.0, 0.5]) ≈ [4.0, -2.0]

    @test cameramap(2.0)                  ≈ LinearMap(UniformScaling(2.0)) ∘ PerspectiveMap()
    @test cameramap((1.1,2.2))            ≈ LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
    @test cameramap(1.1,       (3.3,4.4)) ≈ Translation([-3.3,-4.4]) ∘ LinearMap(UniformScaling(1.1)) ∘ PerspectiveMap()
    @test cameramap((1.1,2.2), (3.3,4.4)) ≈ Translation([-3.3,-4.4]) ∘ LinearMap([1.1 0; 0 2.2]) ∘ PerspectiveMap()
end
