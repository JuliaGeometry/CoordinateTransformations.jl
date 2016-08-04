@testset "Coordinate Systems" begin
    @testset "2D" begin
        c_from_p = CartesianFromPolar()
        p_from_c = PolarFromCartesian()
        identity_c = IdentityTransformation()
        identity_p = IdentityTransformation()

        @test inv(c_from_p) == p_from_c
        @test inv(p_from_c) == c_from_p

        @test p_from_c ∘ c_from_p == identity_p
        @test c_from_p ∘ p_from_c == identity_c

        # test identity
        @test p_from_c ∘ identity_c == p_from_c
        @test identity_p ∘ p_from_c == p_from_c

        # Test all four quadrants of the plane (for consistency of branch-cut)
        # Include derivative tests... compare with automatic differentiation (forward mode from ForwardDiff.Dual)

        # 1st quadrant
        xy = SVector(1.0, 2.0)
        rθ = Polar(2.23606797749979, 1.1071487177940904)
        @test p_from_c(xy) ≈ rθ
        @test p_from_c(collect(xy)) ≈ rθ
        @test c_from_p(rθ) ≈ xy

        # TODO - define some convenience functions to create the gradient numbers and unpack the arrays.
        xy_gn = SVector(Dual(1.0, (1.0,0.0)), Dual(2.0, (0.0,1.0)))
        rθ_gn = p_from_c(xy_gn)
        m_gn = @SMatrix [partials(rθ_gn.r, 1) partials(rθ_gn.r, 2);
                    partials(rθ_gn.θ, 1) partials(rθ_gn.θ, 2) ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(Dual(2.23606797749979, (1.0, 0.0)), Dual(1.1071487177940904, (0.0, 1.0)))
        xy_gn = c_from_p(rθ_gn)
        m_gn = @SMatrix [partials(xy_gn[1], 1) partials(xy_gn[1], 2);
                    partials(xy_gn[2], 1) partials(xy_gn[2], 2) ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 2nd quadrant
        xy = SVector(-1.0, 2.0)
        rθ = Polar(2.23606797749979, 2.0344439357957027)
        @test p_from_c(xy) ≈ rθ
        @test c_from_p(rθ) ≈ xy

        xy_gn = SVector(Dual(-1.0, (1.0,0.0)), Dual(2.0, (0.0,1.0)))
        rθ_gn = p_from_c(xy_gn)
        m_gn = @SMatrix [partials(rθ_gn.r, 1) partials(rθ_gn.r, 2);
                    partials(rθ_gn.θ, 1) partials(rθ_gn.θ, 2) ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(Dual(2.23606797749979, (1.0, 0.0)), Dual(2.0344439357957027, (0.0, 1.0)))
        xy_gn = c_from_p(rθ_gn)
        m_gn = @SMatrix [partials(xy_gn[1], 1) partials(xy_gn[1], 2);
                    partials(xy_gn[2], 1) partials(xy_gn[2], 2) ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 3rd quadrant
        xy = SVector(1.0, -2.0)
        rθ = Polar(2.23606797749979, -1.1071487177940904)
        @test p_from_c(xy) ≈ rθ
        @test c_from_p(rθ) ≈ xy

        xy_gn = SVector(Dual(1.0, (1.0,0.0)), Dual(-2.0, (0.0,1.0)))
        rθ_gn = p_from_c(xy_gn)
        m_gn = @SMatrix [partials(rθ_gn.r, 1) partials(rθ_gn.r, 2);
                    partials(rθ_gn.θ, 1) partials(rθ_gn.θ, 2) ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(Dual(2.23606797749979, (1.0, 0.0)), Dual(-1.1071487177940904, (0.0, 1.0)))
        xy_gn = c_from_p(rθ_gn)
        m_gn = @SMatrix [partials(xy_gn[1], 1) partials(xy_gn[1], 2);
                    partials(xy_gn[2], 1) partials(xy_gn[2], 2) ]
        m = transform_deriv(c_from_p, rθ)
        @test m ≈ m_gn

        # 4th quadrant
        xy = SVector(-1.0, -2.0)
        rθ = Polar(2.23606797749979, -2.0344439357957027)
        @test p_from_c(xy) ≈ rθ
        @test c_from_p(rθ) ≈ xy

        xy_gn = SVector(Dual(-1.0, (1.0,0.0)), Dual(-2.0, (0.0,1.0)))
        rθ_gn = p_from_c(xy_gn)
        m_gn = @SMatrix [partials(rθ_gn.r, 1) partials(rθ_gn.r, 2);
                    partials(rθ_gn.θ, 1) partials(rθ_gn.θ, 2) ]
        m = transform_deriv(p_from_c, xy)
        @test m ≈ m_gn

        rθ_gn = Polar(Dual(2.23606797749979, (1.0, 0.0)), Dual(-2.0344439357957027, (0.0, 1.0)))
        xy_gn = c_from_p(rθ_gn)
        m_gn = @SMatrix [partials(xy_gn[1], 1) partials(xy_gn[1], 2);
                    partials(xy_gn[2], 1) partials(xy_gn[2], 2) ]
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

    end
end
