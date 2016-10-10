@testset "Perspective transformation" begin
    @test PerspectiveMap()([2.0, -1.0, 0.5]) ≈ [4.0, -2.0]

    @test cameramap() === PerspectiveMap()
    @test cameramap(pixel_size = 3e-5, focal_length = 0.3) ≈ LinearMap(UniformScaling(1e4)) ∘ PerspectiveMap()
    @test cameramap(origin = [1.0,2.0,3.0], orientation = [0 1 0; 0 0 1; 1 0 0]) ≈ PerspectiveMap() ∘ AffineMap([0 0 1; 1 0 0; 0 1 0], [-3.0, -1.0, -2.0])
    @test cameramap(offset_x = 100, offset_y = 20) ≈ Translation(SVector(-100, -20)) ∘ PerspectiveMap()
end
