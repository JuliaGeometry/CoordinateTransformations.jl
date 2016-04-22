@testset "Coordinate Systems" begin
    @testset "2D" begin
        T = Point{2, Float64}

        c_from_p = CartesianFromPolar{Float64}()
        p_from_c = PolarFromCartesian{Float64}()
        identity_c = IdentityTransformation{T}()
        identity_p = IdentityTransformation{Polar{Float64}}()

        @test inv(c_from_p) == p_from_c
        @test inv(p_from_c) == c_from_p

        @test p_from_c ∘ c_from_p == identity_p
        @test c_from_p ∘ p_from_c == identity_c

        xy = Point(1.0, 2.0)
        rθ = Polar(2.23606797749979, 1.1071487177940904)

        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        # test all four quadrants
    end

    @testset "3D" begin
        x = Point(1.0, 2.0, 3.0)

    end
end
