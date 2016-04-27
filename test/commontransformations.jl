@testset "Common Transformations" begin
    @testset "Translation" begin
        x = Point(1.0, 2.0)
        trans = Translation(2.0, -1.0)

        # Inverse
        @test inv(trans) == Translation(-2.0, 1.0)

        # Composition
        @test trans ∘ trans == Translation(4.0, -2.0)

        # Transform
        @test transform(trans, x) === Point(3.0, 1.0)

        # Transform derivative
        m1 = transform_deriv(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == eye(2)

        # Transform parameter derivative
        m1 = transform_deriv_params(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == eye(2)
    end

    @testset "RotationPolar" begin
        p = Polar(2.0, 1.0)
        trans = RotationPolar(1.0)

        # Inverse
        @test inv(trans) == RotationPolar(-1.0)

        # Composition
        @test trans ∘ trans == RotationPolar(2.0)

        # Transform
        @test transform(trans, p) == Polar(2.0, 2.0)

        # Transform derivative
        @test transform_deriv(trans, p) == [0 0; 0 1]

        # Transform parameter derivative
        @test transform_deriv_params(trans, p) == [0; 1]
    end


    @testset "Rotation2D" begin
        x = Point(2.0, 0.0)
        x2 = transform(CartesianFromPolar(), Polar(2.0, 1.0))
        trans = Rotation2D(1.0)

        # Constructor
        @test trans.cos == cos(1.0)
        @test trans.sin == sin(1.0)

        # Inverse
        @test inv(trans) == Rotation2D(-1.0)

        # Composition
        @test trans ∘ trans == Rotation2D(2.0)

        # Transform
        @test transform(trans, x) ≈ x2

        # Transform derivative
        x = Point(2.0,1.0)
        x_gn = Point(GradientNumber(2.0, (1.0,0.0)), GradientNumber(1.0, (0.0,1.0)))
        x2_gn = transform(trans, x_gn)
        m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2];
                     x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] ]
        m = transform_deriv(trans, x)
        @test m ≈ m_gn

        # Transform parameter derivative
        trans_gn = Rotation2D(GradientNumber(1.0, (1.0)))
        x = Point(2.0,1.0)
        x2_gn = transform(trans_gn, x)
        m_gn = Mat(x2_gn[1].partials.data[1], x2_gn[2].partials.data[1])
        m = transform_deriv_params(trans, x)
        @test m ≈ m_gn
    end


end
