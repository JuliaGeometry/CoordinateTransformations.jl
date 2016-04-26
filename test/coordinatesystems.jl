@testset "Coordinate Systems" begin
    @testset "2D" begin
        c_from_p = CartesianFromPolar()
        p_from_c = PolarFromCartesian()
        identity_c = IdentityTransformation{Point{2}}()
        identity_p = IdentityTransformation{Polar}()

        @test inv(c_from_p) == p_from_c
        @test inv(p_from_c) == c_from_p

        @test p_from_c ∘ c_from_p == identity_p
        @test c_from_p ∘ p_from_c == identity_c

        # test identity
        @test p_from_c ∘ identity_c == p_from_c
        @test identity_p ∘ p_from_c == p_from_c

        # Test all four quadrants of the plane (for consistency of branch-cut)
        # Include derivative tests... compare with automatic differentiation (forward mode from ForwardDiff.GradientNumber)

        # 1st quadrant
        xy = Point(1.0, 2.0)
        rθ = Polar(2.23606797749979, 1.1071487177940904)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy_gn = Point(GradientNumber(1.0, (1.0,0.0)), GradientNumber(2.0, (0.0,1.0)))
        rθ_gn = transform(p_from_c, xy_gn)
        m_gn = @fsa [rθ_gn.r.partials.data[1] rθ_gn.r.partials.data[2];
                     rθ_gn.θ.partials.data[1] rθ_gn.θ.partials.data[2] ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(GradientNumber(2.23606797749979, (1.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0)))
        xy_gn = transform(c_from_p, rθ_gn)
        m_gn = @fsa [xy_gn[1].partials.data[1] xy_gn[1].partials.data[2];
                     xy_gn[2].partials.data[1] xy_gn[2].partials.data[2] ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 2nd quadrant
        xy = Point(-1.0, 2.0)
        rθ = Polar(2.23606797749979, 2.0344439357957027)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy_gn = Point(GradientNumber(-1.0, (1.0,0.0)), GradientNumber(2.0, (0.0,1.0)))
        rθ_gn = transform(p_from_c, xy_gn)
        m_gn = @fsa [rθ_gn.r.partials.data[1] rθ_gn.r.partials.data[2];
                     rθ_gn.θ.partials.data[1] rθ_gn.θ.partials.data[2] ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(GradientNumber(2.23606797749979, (1.0, 0.0)), GradientNumber(2.0344439357957027, (0.0, 1.0)))
        xy_gn = transform(c_from_p, rθ_gn)
        m_gn = @fsa [xy_gn[1].partials.data[1] xy_gn[1].partials.data[2];
                     xy_gn[2].partials.data[1] xy_gn[2].partials.data[2] ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 3rd quadrant
        xy = Point(1.0, -2.0)
        rθ = Polar(2.23606797749979, -1.1071487177940904)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy_gn = Point(GradientNumber(1.0, (1.0,0.0)), GradientNumber(-2.0, (0.0,1.0)))
        rθ_gn = transform(p_from_c, xy_gn)
        m_gn = @fsa [rθ_gn.r.partials.data[1] rθ_gn.r.partials.data[2];
                     rθ_gn.θ.partials.data[1] rθ_gn.θ.partials.data[2] ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(GradientNumber(2.23606797749979, (1.0, 0.0)), GradientNumber(-1.1071487177940904, (0.0, 1.0)))
        xy_gn = transform(c_from_p, rθ_gn)
        m_gn = @fsa [xy_gn[1].partials.data[1] xy_gn[1].partials.data[2];
                     xy_gn[2].partials.data[1] xy_gn[2].partials.data[2] ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 4th quadrant
        xy = Point(-1.0, -2.0)
        rθ = Polar(2.23606797749979, -2.0344439357957027)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy_gn = Point(GradientNumber(-1.0, (1.0,0.0)), GradientNumber(-2.0, (0.0,1.0)))
        rθ_gn = transform(p_from_c, xy_gn)
        m_gn = @fsa [rθ_gn.r.partials.data[1] rθ_gn.r.partials.data[2];
                     rθ_gn.θ.partials.data[1] rθ_gn.θ.partials.data[2] ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(GradientNumber(2.23606797749979, (1.0, 0.0)), GradientNumber(-2.0344439357957027, (0.0, 1.0)))
        xy_gn = transform(c_from_p, rθ_gn)
        m_gn = @fsa [xy_gn[1].partials.data[1] xy_gn[1].partials.data[2];
                     xy_gn[2].partials.data[1] xy_gn[2].partials.data[2] ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn
    end

    @testset "3D" begin
        s_from_cart = SphericalFromCartesian()
        cart_from_s = CartesianFromSpherical()
        cyl_from_cart = CylindricalFromCartesian()
        cart_from_cyl = CartesianFromCylindrical()
        cyl_from_s = CylindricalFromSpherical()
        s_from_cyl = SphericalFromCylindrical()
        identity_cart = IdentityTransformation{Point{3}}()
        identity_s = IdentityTransformation{Spherical}()
        identity_cyl = IdentityTransformation{Cylindrical}()

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
        xyz = Point(1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 2
        xyz = Point(-1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 3
        xyz = Point(1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(-1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 4
        xyz = Point(-1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(-2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 5
        xyz = Point(1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 6
        xyz = Point(-1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 7
        xyz = Point(1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(-1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Octant 8
        xyz = Point(-1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, -2.0344439357957027, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(-3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cart, xyz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cart, xyz)
        @test m ≈ m_gn

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(-2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(-0.9302740141154721, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_s, rθϕ_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_s, rθϕ)
        @test m ≈ m_gn

        # Cylindrical <-> Cartesian
        # test all 4 quadrants of the xy-plane (for consistency of branch-cuts)

        # First quadrant
        xyz = Point(1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = transform(cyl_from_cart, xyz_gn)
        m_gn = @fsa [rθz_gn.r.partials.data[1] rθz_gn.r.partials.data[2] rθz_gn.r.partials.data[3];
                     rθz_gn.θ.partials.data[1] rθz_gn.θ.partials.data[2] rθz_gn.θ.partials.data[3];
                     rθz_gn.z.partials.data[1] rθz_gn.z.partials.data[2] rθz_gn.z.partials.data[3] ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(GradientNumber(2.23606797749979, (1.0, 0.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_cyl, rθz_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Second quadrant
        xyz = Point(-1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 2.0344439357957027, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = transform(cyl_from_cart, xyz_gn)
        m_gn = @fsa [rθz_gn.r.partials.data[1] rθz_gn.r.partials.data[2] rθz_gn.r.partials.data[3];
                     rθz_gn.θ.partials.data[1] rθz_gn.θ.partials.data[2] rθz_gn.θ.partials.data[3];
                     rθz_gn.z.partials.data[1] rθz_gn.z.partials.data[2] rθz_gn.z.partials.data[3] ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(GradientNumber(2.23606797749979, (1.0, 0.0, 0.0)), GradientNumber(2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_cyl, rθz_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Third quadrant
        xyz = Point(1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, -1.1071487177940904, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz_gn = Point(GradientNumber(1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = transform(cyl_from_cart, xyz_gn)
        m_gn = @fsa [rθz_gn.r.partials.data[1] rθz_gn.r.partials.data[2] rθz_gn.r.partials.data[3];
                     rθz_gn.θ.partials.data[1] rθz_gn.θ.partials.data[2] rθz_gn.θ.partials.data[3];
                     rθz_gn.z.partials.data[1] rθz_gn.z.partials.data[2] rθz_gn.z.partials.data[3] ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(GradientNumber(2.23606797749979, (1.0, 0.0, 0.0)), GradientNumber(-1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_cyl, rθz_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Fourth quadrant
        xyz = Point(-1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, -2.0344439357957027, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz_gn = Point(GradientNumber(-1.0, (1.0, 0.0, 0.0)), GradientNumber(-2.0, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθz_gn = transform(cyl_from_cart, xyz_gn)
        m_gn = @fsa [rθz_gn.r.partials.data[1] rθz_gn.r.partials.data[2] rθz_gn.r.partials.data[3];
                     rθz_gn.θ.partials.data[1] rθz_gn.θ.partials.data[2] rθz_gn.θ.partials.data[3];
                     rθz_gn.z.partials.data[1] rθz_gn.z.partials.data[2] rθz_gn.z.partials.data[3] ]
        m = transform_deriv(cyl_from_cart, xyz)
        @test m ≈ m_gn

        rθz_gn = Cylindrical(GradientNumber(2.23606797749979, (1.0, 0.0, 0.0)), GradientNumber(-2.0344439357957027, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        xyz_gn = transform(cart_from_cyl, rθz_gn)
        m_gn = @fsa [xyz_gn[1].partials.data[1] xyz_gn[1].partials.data[2] xyz_gn[1].partials.data[3];
                     xyz_gn[2].partials.data[1] xyz_gn[2].partials.data[2] xyz_gn[2].partials.data[3];
                     xyz_gn[3].partials.data[1] xyz_gn[3].partials.data[2] xyz_gn[3].partials.data[3] ]
        m = transform_deriv(cart_from_cyl, rθz)
        @test m ≈ m_gn

        # Spherical <-> Cartesian
        # Just composes at the moment, so a single testcase suffices
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test transform(cyl_from_s, rθϕ) ≈ rθz
        @test transform(s_from_cyl, rθz) ≈ rθϕ

        rθϕ_gn = Spherical(GradientNumber(3.7416573867739413, (1.0, 0.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(0.9302740141154721, (0.0, 0.0, 1.0)))
        rθz_gn = transform(cyl_from_s, rθϕ_gn)
        m_gn = @fsa [rθz_gn.r.partials.data[1] rθz_gn.r.partials.data[2] rθz_gn.r.partials.data[3];
                     rθz_gn.θ.partials.data[1] rθz_gn.θ.partials.data[2] rθz_gn.θ.partials.data[3];
                     rθz_gn.z.partials.data[1] rθz_gn.z.partials.data[2] rθz_gn.z.partials.data[3] ]
        m = transform_deriv(cyl_from_s, rθϕ)
        #@test isapprox(m, m_gn; atol = 1e-12)
        for (m1,m2) in zip(m,m_gn) # Unfortunately, FixedSizeArrays doesn't pass the keyword arguments to isapprox...
            @test isapprox(m1, m2; atol=1e-12)
        end

        rθz_gn = Cylindrical(GradientNumber(2.23606797749979, (1.0, 0.0, 0.0)), GradientNumber(1.1071487177940904, (0.0, 1.0, 0.0)), GradientNumber(3.0, (0.0, 0.0, 1.0)))
        rθϕ_gn = transform(s_from_cyl, rθz_gn)
        m_gn = @fsa [rθϕ_gn.r.partials.data[1] rθϕ_gn.r.partials.data[2] rθϕ_gn.r.partials.data[3];
                     rθϕ_gn.θ.partials.data[1] rθϕ_gn.θ.partials.data[2] rθϕ_gn.θ.partials.data[3];
                     rθϕ_gn.ϕ.partials.data[1] rθϕ_gn.ϕ.partials.data[2] rθϕ_gn.ϕ.partials.data[3] ]
        m = transform_deriv(s_from_cyl, rθz)
#        @test isapprox(m, m_gn; atol = 1e-12)
        @test m ≈ m_gn

    end
end
