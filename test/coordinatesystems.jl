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

        # test identity
        @test p_from_c ∘ identity_c == p_from_c
        @test identity_p ∘ p_from_c == p_from_c

        # test all four quadrants of the plane (for consistency of branch-cut)
        xy = Point(1.0, 2.0)
        rθ = Polar(2.23606797749979, 1.1071487177940904)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy = Point(-1.0, 2.0)
        rθ = Polar(2.23606797749979, 2.0344439357957027)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy = Point(1.0, -2.0)
        rθ = Polar(2.23606797749979, -1.1071487177940904)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy

        xy = Point(-1.0, -2.0)
        rθ = Polar(2.23606797749979, 4.2487413713838835)
        @test transform(p_from_c, xy) ≈ rθ
        @test transform(c_from_p, rθ) ≈ xy
    end

    @testset "3D" begin
        T = Point{3, Float64}
        s_from_cart = SphericalFromCartesian{Float64}()
        cart_from_s = CartesianFromSpherical{Float64}()
        cyl_from_cart = CylindricalFromCartesian{Float64}()
        cart_from_cyl = CartesianFromCylindrical{Float64}()
        cyl_from_s = CylindricalFromSpherical{Float64}()
        s_from_cyl = SphericalFromCylindrical{Float64}()
        identity_cart = IdentityTransformation{T}()
        identity_s = IdentityTransformation{Spherical{Float64}}()
        identity_cyl = IdentityTransformation{Cylindrical{Float64}}()

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
        xyz = Point(1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(-1.0, 2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(-1.0, -2.0, 3.0)
        rθϕ = Spherical(3.7416573867739413, 4.2487413713838835, 0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(-1.0, 2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 2.0344439357957027, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, -1.1071487177940904, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        xyz = Point(-1.0, -2.0, -3.0)
        rθϕ = Spherical(3.7416573867739413, 4.2487413713838835, -0.9302740141154721)
        @test transform(s_from_cart, xyz) ≈ rθϕ
        @test transform(cart_from_s, rθϕ) ≈ xyz

        # Cylindrical <-> Cartesian
        # test all 4 quadrants of the xy-plane (for consistency of branch-cuts)
        xyz = Point(1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz = Point(-1.0, 2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 2.0344439357957027, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz = Point(1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, -1.1071487177940904, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        xyz = Point(-1.0, -2.0, 3.0)
        rθz = Cylindrical(2.23606797749979, 4.2487413713838835, 3.0)
        @test transform(cyl_from_cart, xyz) ≈ rθz
        @test transform(cart_from_cyl, rθz) ≈ xyz

        # Spherical <-> Cartesian
        # Just composes at the moment, so a single test suffices
        rθϕ = Spherical(3.7416573867739413, 1.1071487177940904, 0.9302740141154721)
        rθz = Cylindrical(2.23606797749979, 1.1071487177940904, 3.0)
        @test transform(cyl_from_s, rθϕ) ≈ rθz
        @test transform(s_from_cyl, rθz) ≈ rθϕ
    end
end
