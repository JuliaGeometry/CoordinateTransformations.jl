
# A transformation for testing the local affine mapping
struct SquareMe <: Transformation; end
(::SquareMe)(x) = x.^2
CoordinateTransformations.transform_deriv(::SquareMe, x0) = Matrix(Diagonal(2*x0))

struct Ray37
    origin::Vector
    direction::Vector
end
function (m::LinearMap)(r::Ray37)
    Ray37(m(r.origin),
          m(r.direction))
end
function (t::Translation)(r::Ray37)
    Ray37(t(r.origin),
          r.direction)
end

@testset "Common Transformations" begin
    @testset "AffineMap" begin
        @testset "Simple" begin
            M = [1 2; 3 4]
            v = [-1, 1]
            x = [1,0]
            y = [0.0,1.0]
            A = AffineMap(M,v)
            @test A(x) == M*x + v
        end

        @testset "composition " begin
            M1 = [1 2; 3 4]
            v1 = [-1, 1]
            A1 = AffineMap(M1,v1)
            M2 = [0 1; 1 0]
            v2 = [-2, 0]
            A2 = AffineMap(M2,v2)
            x = [1,0]
            y = [0,1]
            @test A1(x) == M1*x + v1
            @test A1(y) == M1*y + v1
            @test (A2∘A1)(x) == M2*(M1*x + v1) + v2
            @test (A2∘A1)(y) == M2*(M1*y + v1) + v2
        end

        @testset "inverse" begin
            M = [1.0 2.0; 3.0 4.0]
            v = [-1.0, 1.0]
            x = [1,0]
            y = [0.0,1.0]
            A = AffineMap(M,v)
            @test inv(A)(A(x)) ≈ x
            @test inv(A)(A(y)) ≈ y
        end

        @testset "Affine approximation" begin
            S = SquareMe()
            x0 = [1,2,3]
            dx = 0.1*[1,-1,1]
            A = AffineMap(S, x0)
            @test isapprox(S(x0 + dx), A(x0 + dx), atol=maximum(2*dx.^2))
        end
    end

    @testset "LinearMap" begin
        M = [1 2; 3 4]
        x = [1,0]
        y = [0,1]
        L = LinearMap(M)
        @test L(x) == M*x
        @test inv(L)(x) == inv(M)*x
        @test inv(L)(y) == inv(M)*y
        @test inv(L)(L(x)) ≈ x
        @test inv(L)(L(y)) ≈ y
        @test (L∘L)(x) == (M*M)*x
    end

    @testset "Translation" begin
        x = SVector(1.0, 2.0)
        trans = Translation(2.0, -1.0)

        # Inverse
        @test inv(trans) == Translation(-2.0, 1.0)

        # Composition
        @test trans ∘ trans == Translation(4.0, -2.0)

        # Transform
        @test trans(x) === SVector(3.0, 1.0)
        @test trans(collect(x)) == [3.0, 1.0]

        # Transform derivative
        m1 = transform_deriv(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == Matrix(I, 2, 2)  # in v0.7 and above, this can just be `== I`

        # Transform parameter derivative
        m1 = transform_deriv_params(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == Matrix(I, 2, 2)  # In v0.7 and above, this can just be `== I`
    end

    @testset "Recenter" begin
        M = [1 2; 3 4]
        for origin in ([5,-3], @SVector([5,-3]))
            c = recenter(M, origin)
            @test c(origin) == origin
            @test c(zero(origin)) == [6,-6]
        end

        # Tuple is converted to SVector first
        origin = (5, -3)
        new_origin = SVector(origin)
        c = recenter(M, origin)
        @test c(origin) == new_origin
        @test c(zero(new_origin)) == [6, -6]
    end

    @testset "application of AffineMap in terms of LinearMap and Translation" begin
        origin = randn(3)
        direction = randn(3)
        linear = randn(3,3)
        trans = randn(3)
        r = Ray37(origin, direction)
        expected = Ray37(linear*origin+trans, linear*direction)
        result   = AffineMap(linear, trans)(r)
        @test expected.origin ≈ result.origin
        @test expected.direction ≈ result.direction
    end
end
