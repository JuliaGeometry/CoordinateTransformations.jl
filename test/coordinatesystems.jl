@testset "Coordinate Systems" begin
    function jacobian(xy::AbstractVector)
        SA[ partials(xy[1], 1) partials(xy[1], 2);
            partials(xy[2], 1) partials(xy[2], 2) ]
    end

    function jacobian(rθ::AbstractPolar)
        SA[ partials(rθ.r, 1) partials(rθ.r, 2);
            partials(rθ.θ, 1) partials(rθ.θ, 2) ]
    end

    get_PT(::Polar) = Polar
    get_PT(::Polard) = Polard

    function test_jacobian(xy, rθ, tform)
        # create dual vector
        xy_gn = SA[Dual(xy[1], (1.0, 0.0)), Dual(xy[2], (0.0, 1.0))]

        # forward transform
        rθ_gn = tform(xy_gn)
        m_gn = jacobian(rθ_gn)
        m = transform_deriv(tform, xy)
        @test m ≈ m_gn

        # inverse transform
        rθ_gn = get_PT(rθ)(Dual(rθ.r, (1.0, 0.0)), Dual(rθ.θ, (0.0, 1.0)))
        xy_gn = inv(tform)(rθ_gn)
        m_gn = jacobian(xy_gn)
        m = transform_deriv(inv(tform), rθ)
        @test m ≈ m_gn
    end

    @testset "2D" begin
        c_from_p = CartesianFromPolar()
        p_from_c = PolarFromCartesian()
        @test p_from_c isa PolarFromCartesian{Polar} # test defaults to Polar
        pd_from_c = PolardFromCartesian()
        @test pd_from_c isa PolarFromCartesian{Polard}
        identity_c = IdentityTransformation()
        identity_p = IdentityTransformation()

        @test inv(c_from_p) == p_from_c
        @test inv(p_from_c) == c_from_p
        @test inv(pd_from_c) == c_from_p

        @test p_from_c ∘ c_from_p == identity_p
        @test c_from_p ∘ p_from_c == identity_c

        @test pd_from_c ∘ c_from_p == identity_p
        @test c_from_p ∘ pd_from_c == identity_c

        # test identity
        @test p_from_c ∘ identity_c == p_from_c
        @test pd_from_c ∘ identity_c == pd_from_c
        @test identity_p ∘ p_from_c == p_from_c
        @test identity_p ∘ pd_from_c == pd_from_c

        # Test all four quadrants of the plane (for consistency of branch-cut)
        # Include derivative tests... compare with automatic differentiation (forward mode from ForwardDiff.Dual)

        # 1st quadrant
        xy = SVector(1.0, 2.0)
        rθ = Polar(2.23606797749979, 1.1071487177940904)
        rθd = Polard(2.23606797749979, rad2deg(1.1071487177940904))
        @test p_from_c(xy) ≈ rθ
        @test pd_from_c(xy) ≈ rθd
        @test p_from_c(collect(xy)) ≈ rθ
        @test pd_from_c(collect(xy)) ≈ rθd
        @test c_from_p(rθ) ≈ xy
        @test c_from_p(rθd) ≈ xy

        @test rθ ≈ rθd
        @test CoordinateTransformations.angle(rθ) ≈ CoordinateTransformations.angle(rθd)

        test_jacobian(xy, rθ, p_from_c)
        test_jacobian(xy, rθd, pd_from_c)


        # 2nd quadrant
        xy = SVector(-1.0, 2.0)
        rθ = Polar(2.23606797749979, 2.0344439357957027)
        rθd = Polard(2.23606797749979, rad2deg(2.0344439357957027))
        @test p_from_c(xy) ≈ rθ
        @test pd_from_c(xy) ≈ rθd
        @test p_from_c(collect(xy)) ≈ rθ
        @test pd_from_c(collect(xy)) ≈ rθd
        @test c_from_p(rθ) ≈ xy
        @test c_from_p(rθd) ≈ xy

        test_jacobian(xy, rθ, p_from_c)
        test_jacobian(xy, rθd, pd_from_c)

        # 3rd quadrant
        xy = SVector(1.0, -2.0)
        rθ = Polar(2.23606797749979, -1.1071487177940904)
        rθd = Polard(2.23606797749979, rad2deg(-1.1071487177940904))
        @test p_from_c(xy) ≈ rθ
        @test pd_from_c(xy) ≈ rθd
        @test p_from_c(collect(xy)) ≈ rθ
        @test pd_from_c(collect(xy)) ≈ rθd
        @test c_from_p(rθ) ≈ xy
        @test c_from_p(rθd) ≈ xy

        test_jacobian(xy, rθ, p_from_c)
        test_jacobian(xy, rθd, pd_from_c)

        # 4th quadrant
        xy = SVector(-1.0, -2.0)
        rθ = Polar(2.23606797749979, -2.0344439357957027)
        rθd = Polard(2.23606797749979, rad2deg(-2.0344439357957027))
        @test p_from_c(xy) ≈ rθ
        @test pd_from_c(xy) ≈ rθd
        @test p_from_c(collect(xy)) ≈ rθ
        @test pd_from_c(collect(xy)) ≈ rθd
        @test c_from_p(rθ) ≈ xy
        @test c_from_p(rθd) ≈ xy

        test_jacobian(xy, rθ, p_from_c)
        test_jacobian(xy, rθd, pd_from_c)

        @testset "Common types - Polar" begin
            xy = SVector(1.0, 2.0)
            xy_i = SVector(1,2)
            p1 = Polar(1, 2.0f0)
            p2 = Polar(1.0, 2)
            p3 = Polar{Int, Float64}(1, 2.0)
            rθ = Polar(2.23606797749979, 1.1071487177940904)

            @test typeof(p1.r) == typeof(p1.θ)
            @test typeof(p2.r) == typeof(p2.θ)
            @test typeof(p3.r) == Int
            @test typeof(p3.θ) == Float64

            @test p_from_c(xy_i) ≈ rθ
            @test p_from_c(xy) ≈ rθ
            @test p_from_c(collect(xy)) ≈ rθ
            @test c_from_p(rθ) ≈ xy
        end

        @testset "Common types - Polard" begin
            xy = SVector(1.0, 2.0)
            xy_i = SVector(1,2)
            p1 = Polard(1, rad2deg(2.0f0))
            p2 = Polard(1.0, rad2deg(2))
            p3 = Polard{Int, Float64}(1, rad2deg(2.0))
            rθ = Polard(2.23606797749979, rad2deg(1.1071487177940904))

            @test typeof(p1.r) == typeof(p1.θ)
            @test typeof(p2.r) == typeof(p2.θ)
            @test typeof(p3.r) == Int
            @test typeof(p3.θ) == Float64

            @test pd_from_c(xy_i) ≈ rθ
            @test pd_from_c(xy) ≈ rθ
            @test pd_from_c(collect(xy)) ≈ rθ
            @test c_from_p(rθ) ≈ xy
        end

        @testset "Units - Polar" begin
            xy = SVector(1.0, 2.0)u"m"
            rθ = Polar(2.23606797749979u"m", 1.1071487177940904)

            @test_broken p_from_c(xy) ≈ rθ
            @test_broken p_from_c(collect(xy)) ≈ rθ
            @test c_from_p(rθ) ≈ xy
        end

        @testset "Units - Polard" begin
            xy = SVector(1.0, 2.0)u"m"
            rθ = Polard(2.23606797749979u"m", rad2deg(1.1071487177940904))

            @test pd_from_c(xy) ≈ rθ
            @test pd_from_c(collect(xy)) ≈ rθ
            @test c_from_p(rθ) ≈ xy
        end
    end

    @testset "3D" begin
        s_from_cart = SphericalFromCartesian()
        cart_from_s = CartesianFromSpherical()
        cyl_from_cart = CylindricalFromCartesian()
        cart_from_cyl = CartesianFromCylindrical()
        cyl_from_s = CylindricalFromSpherical()
        s_from_cyl = SphericalFromCylindrical()
        identity_cart = IdentityTransformation()
        identity_s = IdentityTransformation()
        identity_cyl = IdentityTransformation()

        # inverses
        @test inv(s_from_cart) == cart_from_s
        @test inv(cart_from_s) == s_from_cart
        @test inv(cyl_from_cart) == cart_from_cyl
        @test inv(cart_from_cyl) == cyl_from_cart
        @test inv(s_from_cyl) == cyl_from_s
        @test inv(cyl_from_s) == s_from_cyl

        # composition of inverses
        @test s_from_cart ∘ cart_from_s == identity_s
        @test cart_from_s ∘ s_from_cart == identity_cart
        @test cyl_from_cart ∘ cart_from_cyl == identity_cyl
        @test cart_from_cyl ∘ cyl_from_cart == identity_cart
        @test s_from_cyl ∘ cyl_from_s == identity_s
        @test cyl_from_s ∘ s_from_cyl == identity_cyl

        # cyclic composition
        @test s_from_cart ∘ cart_from_cyl == s_from_cyl
        @test cart_from_s ∘ s_from_cyl == cart_from_cyl
        @test cyl_from_cart ∘ cart_from_s == cyl_from_s
        @test cart_from_cyl ∘ cyl_from_s == cart_from_s
        @test s_from_cyl ∘ cyl_from_cart == s_from_cart
        @test cyl_from_s ∘ s_from_cart == cyl_from_cart

        # Spherical <-> Cartesian
        # test all 8 octants of the sphere (for consistency of branch-cuts)

        # Octant 1
        xyz = SVector(1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test s_from_cart(collect(xyz)) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 2
        xyz = SVector(-1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, 0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 3
        xyz = SVector(1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, 0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 4
        xyz = SVector(-1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, 0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 5
        xyz = SVector(1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, -0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 6
        xyz = SVector(-1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, -0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 7
        xyz = SVector(1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, -0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 8
        xyz = SVector(-1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, -0.9302740141154721)
        @test s_from_cart(xyz) ≈ rθϕ
        @test cart_from_s(rθϕ) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0, 0.0)), Dual(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Cylindrical <-> Cartesian
        # test all 4 quadrants of the xy-plane (for consistency of branch-cuts)

        # First quadrant
        xyz = SVector(1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test cyl_from_cart(xyz) ≈ rθz
        @test cyl_from_cart(collect(xyz)) ≈ rθz
        @test cart_from_cyl(rθz) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = cyl_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                     partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                     partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_cyl(rθz_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Second quadrant
        xyz = SVector(-1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 2.0344439357957027, 3.0)
        @test cyl_from_cart(xyz) ≈ rθz
        @test cart_from_cyl(rθz) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = cyl_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                     partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                     partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_cyl(rθz_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Third quadrant
        xyz = SVector(1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, -1.1071487177940904, 3.0)
        @test cyl_from_cart(xyz) ≈ rθz
        @test cart_from_cyl(rθz) ≈ xyz

        xyz_gn = SVector(Dual(1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = cyl_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                     partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                     partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_cyl(rθz_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Fourth quadrant
        xyz = SVector(-1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, -2.0344439357957027, 3.0)
        @test cyl_from_cart(xyz) ≈ rθz
        @test cart_from_cyl(rθz) ≈ xyz

        xyz_gn = SVector(Dual(-1.0, (1.0, 0.0, 0.0)), Dual(-2.0, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = cyl_from_cart(xyz_gn)
        m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                     partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                     partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = cart_from_cyl(rθz_gn)
        m_gn = @SMatrix [partials(xyz_gn[1], 1) partials(xyz_gn[1], 2) partials(xyz_gn[1], 3);
                     partials(xyz_gn[2], 1) partials(xyz_gn[2], 2) partials(xyz_gn[2], 3);
                     partials(xyz_gn[3], 1) partials(xyz_gn[3], 2) partials(xyz_gn[3], 3) ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Spherical <-> Cartesian
        # Just composes at the moment, so a single testcase suffices
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test cyl_from_s(rθϕ) ≈ rθz
        @test s_from_cyl(rθz) ≈ rθϕ

        rθϕ_gn = Spherical(Dual(3.7416573867739413, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(0.9302740141154721, (0.0, 0.0, 1.0)))
        rθz_gn = cyl_from_s(rθϕ_gn)
        m_gn = @SMatrix [partials(rθz_gn.r, 1) partials(rθz_gn.r, 2) partials(rθz_gn.r, 3);
                     partials(rθz_gn.θ, 1) partials(rθz_gn.θ, 2) partials(rθz_gn.θ, 3);
                     partials(rθz_gn.z, 1) partials(rθz_gn.z, 2) partials(rθz_gn.z, 3) ]
        m = transform_deriv(cyl_from_s, rθϕ)
        #@test isapprox(m, m_gn; atol = 1e-12)
        for (m1,m2) in zip(m,m_gn) # Unfortunately, FixedSizeArrays doesn't pass the keyword arguments to isapprox...
            @test isapprox(m1, m2; atol=1e-12)
        end

        rθz_gn = Cylindrical(Dual(2.23606797749979, (1.0, 0.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0, 0.0)), Dual(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = s_from_cyl(rθz_gn)
        m_gn = @SMatrix [partials(rθϕ_gn.r, 1) partials(rθϕ_gn.r, 2) partials(rθϕ_gn.r, 3);
                     partials(rθϕ_gn.θ, 1) partials(rθϕ_gn.θ, 2) partials(rθϕ_gn.θ, 3);
                     partials(rθϕ_gn.ϕ, 1) partials(rθϕ_gn.ϕ, 2) partials(rθϕ_gn.ϕ, 3) ]
        m = transform_deriv(s_from_cyl, rθz)
#        @test isapprox(m, m_gn; atol = 1e-12)
        @test m ≈ m_gn

        @testset "Common types" begin
            xyz = SVector(1.0, 2.0, 3.0)
            xyz_i = SVector(1, 2, 3)

            @testset "Spherical" begin
                rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)

                @test s_from_cart(xyz) ≈ rθϕ
                @test s_from_cart(xyz_i) ≈ rθϕ
                @test s_from_cart(collect(xyz)) ≈ rθϕ
                @test cart_from_s(rθϕ) ≈ xyz

                s1 = Spherical(1, 2.0, 3.0)
                s2 = Spherical(1.0, 2, 3)
                s3 = Spherical{Int,Int}(1, 2, 3)

                @test typeof(s1.r) == typeof(s1.θ) == typeof(s1.ϕ) == Float64
                @test typeof(s2.r) == typeof(s2.θ) == typeof(s2.ϕ) == Float64
                @test typeof(s3.r) == typeof(s3.θ) == typeof(s3.ϕ) == Int
            end

            @testset "Cylindrical" begin
                rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)

                @test cyl_from_cart(xyz) ≈ rθz
                @test cyl_from_cart(xyz_i) ≈ rθz
                @test cyl_from_cart(collect(xyz)) ≈ rθz
                @test cart_from_cyl(rθz) ≈ xyz

                c1 = Cylindrical(1, 2.0, 3)
                c2 = Cylindrical(1.0, 2, 3.0)
                c3 = Cylindrical(1, 2, 3)
                c4 = Cylindrical{Int,Int}(1, 2, 3)

                @test typeof(c1.r) == typeof(c1.z) == typeof(c1.θ) == Float64
                @test typeof(c2.r) == typeof(c2.θ) == typeof(c2.z) == Float64
                @test typeof(cyl_from_cart(xyz_i).r) == typeof(cyl_from_cart(xyz_i).z) == Float64
                @test c3 == c4
            end
        end

        @testset "Units" begin
            xyz = SVector(1.0, 2.0, 3.0)u"m"

            @testset "Shperical" begin
                rθϕ = Spherical(3.7416573867739413u"m", 1.1071487177940904, 0.9302740141154721)

                @test_broken s_from_cart(xyz) ≈ rθϕ
                @test typeof(s_from_cart(xyz)) == typeof(rθϕ)
                @test_broken s_from_cart(collect(xyz)) ≈ rθϕ
                @test cart_from_s(rθϕ) ≈ xyz
            end
            @testset "Cylindrical" begin
                rθz = Cylindrical(2.23606797749979u"m", 1.1071487177940904, 3.0u"m")

                @test_broken cyl_from_cart(xyz) ≈ rθz
                @test typeof(cyl_from_cart(xyz)) == typeof(rθz)
                @test_broken cyl_from_cart(collect(xyz)) ≈ rθz
                @test cart_from_cyl(rθz) ≈ xyz
            end
        end
    end
end
