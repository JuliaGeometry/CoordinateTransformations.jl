immutable SquareMe <: Transformation; end

@compat (::SquareMe)(x) = x.^2
CoordinateTransformations.transform_deriv(::SquareMe, x0) = diagm(2*x0)


@testset "Common Transformations" begin
    @testset "AffineTransformation" begin
        S = SquareMe()
        x0 = [1,2,3]
        dx = 0.1*[1,-1,1]
        A = AffineTransformation(S, x0)
        @test isapprox(S(x0 + dx), A(x0 + dx), atol=maximum(2*dx.^2))
    end
    @testset "Translation" begin
        x = Point(1.0, 2.0)
        trans = Translation(2.0, -1.0)

        # Inverse
        @test inv(trans) == Translation(-2.0, 1.0)

        # Composition
        @test trans ∘ trans == Translation(4.0, -2.0)

        # Transform
        @test trans(x) === Point(3.0, 1.0)
        @test trans(collect(x)) === Vec(3.0, 1.0)

        # Transform derivative
        m1 = transform_deriv(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == eye(2)

        # Transform parameter derivative
        m1 = transform_deriv_params(trans, x)
        m2 = [m1[i,j] for i=1:2, j=1:2] # m1 might be a UniformScaling
        @test m2 == eye(2)
    end

    @testset "Rotation2D on Polar" begin
        p = Polar(2.0, 1.0)
        trans = Rotation2D(1.0)

        # Inverse
        @test inv(trans) == Rotation2D(-1.0)

        # Composition
        @test trans ∘ trans == Rotation2D(2.0)

        # Transform
        @test trans(p) == Polar(2.0, 2.0)

        # Transform derivative
        @test transform_deriv(trans, p) == [0 0; 0 1]

        # Transform parameter derivative
        @test transform_deriv_params(trans, p) == [0; 1]
    end


    @testset "Rotation2D" begin
        x = Point(2.0, 0.0)
        x2 = CartesianFromPolar()(Polar(2.0, 1.0))
        trans = Rotation2D(1.0)

        # Constructor
        @test trans.cos == cos(1.0)
        @test trans.sin == sin(1.0)

        # Inverse
        @test inv(trans) == Rotation2D(-1.0)

        # Composition
        @test trans ∘ trans == Rotation2D(2.0)

        # Transform
        @test trans(x) ≈ x2
        @test Vec(trans(Tuple(x))) ≈ x2
        @test trans(collect(x)) ≈ collect(x2)

        # Transform derivative
        x = Point(2.0,1.0)
        x_gn = Point(Dual(2.0, (1.0,0.0)), Dual(1.0, (0.0,1.0)))
        x2_gn = trans(x_gn)
        m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2);
                     partials(x2_gn[2], 1) partials(x2_gn[2], 2) ]
        m = transform_deriv(trans, x)
        @test m ≈ m_gn

        # Transform parameter derivative
        trans_gn = Rotation2D(Dual(1.0, (1.0)))
        x = Point(2.0,1.0)
        x2_gn = trans_gn(x)
        m_gn = Mat(partials(x2_gn[1], 1), partials(x2_gn[2], 1))
        m = transform_deriv_params(trans, x)
        @test m ≈ m_gn
    end

    @testset "Rotation (3D)" begin

        @testset "Rotation{Void} (rotation matrix parameterization)" begin
            θx = 0.1
            θy = 0.2
            θz = 0.3

            sx = sin(θx)
            cx = cos(θx)
            sy = sin(θy)
            cy = cos(θy)
            sz = sin(θz)
            cz = cos(θz)

            Rx = [ cx  sx  0;
                  -sx  cx  0;
                    0   0  1]

            Ry = [1   0  0 ;
                  0  cy -sy;
                  0  sy  cy]

            Rz = [cx  0 -sx;
                  0   1  0 ;
                  sx  0  cx]

            R = Mat{3,3,Float64}(Rx*Ry*Rz)

            x = Point(1.0, 2.0, 3.0)
            trans = Rotation(R)

            @test inv(trans).matrix ≈ R'
            @test (trans ∘ trans).matrix ≈ R*R

            y = trans(x)
            @test y == R * Vec(1.0, 2.0, 3.0)
            @test trans(Tuple(x)) == Tuple(R * Vec(1.0, 2.0, 3.0))
            @test trans(collect(x)) == R * Vec(1.0, 2.0, 3.0)


            x_gn = Point(Dual(1.0,(1.,0.,0.)), Dual(2.0,(0.,1.,0.)), Dual(3.0,(0.,0.,1.)))
            y_gn = trans(x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[partials(y_gn[i], j) for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            g11 = Dual(R[1,1],(1.,0.,0.,0.,0.,0.,0.,0.,0.))
            g12 = Dual(R[1,2],(0.,1.,0.,0.,0.,0.,0.,0.,0.))
            g13 = Dual(R[1,3],(0.,0.,1.,0.,0.,0.,0.,0.,0.))
            g21 = Dual(R[2,1],(0.,0.,0.,1.,0.,0.,0.,0.,0.))
            g22 = Dual(R[2,2],(0.,0.,0.,0.,1.,0.,0.,0.,0.))
            g23 = Dual(R[2,3],(0.,0.,0.,0.,0.,1.,0.,0.,0.))
            g31 = Dual(R[3,1],(0.,0.,0.,0.,0.,0.,1.,0.,0.))
            g32 = Dual(R[3,2],(0.,0.,0.,0.,0.,0.,0.,1.,0.))
            g33 = Dual(R[3,3],(0.,0.,0.,0.,0.,0.,0.,0.,1.))

            G = @fsa [g11 g12 g13;
                      g21 g22 g23;
                      g31 g32 g33 ]

            trans_gn = Rotation(G)
            y_gn = trans_gn(x)

            M_gn = Mat{3,9,Float64}(vcat(ntuple(i->[partials(y_gn[i], j) for j = 1:9].', 3)...))
            M = transform_deriv_params(trans, x)
            @test M ≈ M_gn
        end

        @testset "Rotation{Quaternion} (quaternion parameterization)" begin
            v = [0.35, 0.45, 0.25, 0.15]
            v = v / vecnorm(v)
            q = Quaternion(v[1],v[2],v[3],v[4],true)

            trans = Rotation(q)
            x = Point(1.0, 2.0, 3.0)

            @test inv(trans) ≈ Rotation(inv(Quaternion(v[1],v[2],v[3],v[4])))
            @test trans ∘ trans ≈ Rotation(trans.matrix * trans.matrix)

            y = trans(x)
            @test y ≈ Point(3.439024390243902,-1.1463414634146332,0.9268292682926829)

            x_gn = Point(Dual(1.0,(1.,0.,0.)), Dual(2.0,(0.,1.,0.)), Dual(3.0,(0.,0.,1.)))
            y_gn = trans(x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[partials(y_gn[i], j) for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            v_gn = [Dual(v[1],(1.,0.,0.,0.)), Dual(v[2],(0.,1.,0.,0.)), Dual(v[3],(0.,0.,1.,0.)), Dual(v[4],(0.,0.,0.,1.))]
            q_gn = Quaternion(v_gn[1],v_gn[2],v_gn[3],v_gn[4],true)
            trans_gn = Rotation(q_gn)
            y_gn = trans_gn(x)
            M_gn = Mat{3,4,Float64}(vcat(ntuple(i->[partials(y_gn[i], j) for j = 1:4].', 3)...))
            M = transform_deriv_params(trans, x)
            # Project both to tangent plane of normalized quaternions (there seems to be a change in definition...)
            proj = Mat{4,4,Float64}(eye(4) - v*v')
            @test M*proj ≈ M_gn*proj
        end

        @testset "Rotation{EulerAngles} (Euler angle parameterization)" begin
            θx = 0.1
            θy = 0.2
            θz = 0.3

            trans = Rotation(EulerAngles(θx, θy, θz))
            x = Point(1.0, 2.0, 3.0)

            @test inv(trans) == Rotation(trans.matrix')
            @test trans ∘ trans == Rotation(trans.matrix * trans.matrix)

            y = trans(x)
            @test y == Point(0.9984766744283545,2.1054173473736495,2.92750100324502)

            x_gn = Point(Dual(1.0,(1.,0.,0.)), Dual(2.0,(0.,1.,0.)), Dual(3.0,(0.,0.,1.)))
            y_gn = trans(x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[partials(y_gn[i], j) for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            # Parameter derivative not defined
            # TODO?
        end

        @testset "RotationXY, RotationYZ and RotationZX" begin
            # RotationXY
            x = Point(2.0, 0.0, 0.0)
            x2 = CartesianFromSpherical()(Spherical(2.0, 1.0, 0.0))
            trans = RotationXY(1.0)

            # Constructor
            @test trans.cos == cos(1.0)
            @test trans.sin == sin(1.0)

            # Inverse
            @test inv(trans) == RotationXY(-1.0)
            @test RotationYX(1.0) == RotationXY(-1.0)

            # Composition
            @test trans ∘ trans == RotationXY(2.0)

            # Transform
            @test trans(x) ≈ x2
            @test Vec(trans(Tuple(x))) ≈ x2
            @test Vec(trans(collect(x))) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(Dual(2.0, (1.0,0.0,0.0)), Dual(1.0, (0.0,1.0,0.0)), Dual(3.0, (0.0,0.0,1.0)))
            x2_gn = trans(x_gn)
            m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2) partials(x2_gn[1], 3);
                         partials(x2_gn[2], 1) partials(x2_gn[2], 2) partials(x2_gn[2], 3);
                         partials(x2_gn[3], 1) partials(x2_gn[3], 2) partials(x2_gn[3], 3) ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationXY(Dual(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = trans_gn(x)
            m_gn = Mat(partials(x2_gn[1], 1), partials(x2_gn[2], 1), partials(x2_gn[3], 1))
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn


            # RotationYZ
            x = Point(0.0, 2.0, 0.0)
            x2 = CartesianFromSpherical()(Spherical(2.0, pi/2, 1.0))
            trans = RotationYZ(1.0)

            # Constructor
            @test trans.cos == cos(1.0)
            @test trans.sin == sin(1.0)

            # Inverse
            @test inv(trans) == RotationYZ(-1.0)
            @test RotationZY(1.0) == RotationYZ(-1.0)

            # Composition
            @test trans ∘ trans == RotationYZ(2.0)

            # Transform
            @test trans(x) ≈ x2
            @test Vec(trans(Tuple(x))) ≈ x2
            @test Vec(trans(collect(x))) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(Dual(2.0, (1.0,0.0,0.0)), Dual(1.0, (0.0,1.0,0.0)), Dual(3.0, (0.0,0.0,1.0)))
            x2_gn = trans(x_gn)
            m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2) partials(x2_gn[1], 3);
                         partials(x2_gn[2], 1) partials(x2_gn[2], 2) partials(x2_gn[2], 3);
                         partials(x2_gn[3], 1) partials(x2_gn[3], 2) partials(x2_gn[3], 3) ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationYZ(Dual(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = trans_gn(x)
            m_gn = Mat(partials(x2_gn[1], 1), partials(x2_gn[2], 1), partials(x2_gn[3], 1))
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn


            # RotationZX
            x = Point(2.0, 0.0, 0.0)
            x2 = CartesianFromSpherical()(Spherical(2.0, 0.0, -1.0))
            trans = RotationZX(1.0)

            # Constructor
            @test trans.cos == cos(1.0)
            @test trans.sin == sin(1.0)

            # Inverse
            @test inv(trans) == RotationZX(-1.0)
            @test RotationXZ(1.0) == RotationZX(-1.0)

            # Composition
            @test trans ∘ trans == RotationZX(2.0)

            # Transform
            @test trans(x) ≈ x2
            @test Vec(trans(Tuple(x))) ≈ x2
            @test Vec(trans(collect(x))) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(Dual(2.0, (1.0,0.0,0.0)), Dual(1.0, (0.0,1.0,0.0)), Dual(3.0, (0.0,0.0,1.0)))
            x2_gn = trans(x_gn)
            m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2) partials(x2_gn[1], 3);
                         partials(x2_gn[2], 1) partials(x2_gn[2], 2) partials(x2_gn[2], 3);
                         partials(x2_gn[3], 1) partials(x2_gn[3], 2) partials(x2_gn[3], 3) ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationZX(Dual(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = trans_gn(x)
            m_gn = Mat(partials(x2_gn[1], 1), partials(x2_gn[2], 1), partials(x2_gn[3], 1))
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn
        end

        @testset "euler_rotation() and composed derivatives" begin
            x = Point(2.0,1.0,3.0)
            trans = euler_rotation(0.1,0.2,0.3)
            x2 = Point(2.730537054338937,0.8047190852558106,2.428290466296628)

            #@test trans.t1.t1 == RotationXY(0.1)
            #@test trans.t1.t2 == RotationYZ(0.2)
            #@test trans.t2 == RotationZX(0.3)

            @test inv(trans) ≈ RotationZX(-0.3) ∘ (RotationYZ(-0.2) ∘ RotationXY(-0.1))

            @test trans(x) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(Dual(2.0, (1.0,0.0,0.0)), Dual(1.0, (0.0,1.0,0.0)), Dual(3.0, (0.0,0.0,1.0)))
            x2_gn = trans(x_gn)
            m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2) partials(x2_gn[1], 3);
                         partials(x2_gn[2], 1) partials(x2_gn[2], 2) partials(x2_gn[2], 3);
                         partials(x2_gn[3], 1) partials(x2_gn[3], 2) partials(x2_gn[3], 3) ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative

            #trans_gn = euler_rotation(Dual(0.1, (1.0, 0.0, 0.0)), Dual(0.2, (0.0, 1.0, 0.0)), Dual(0.3, (0.0, 0.0, 1.0)))
            #x = Point(2.0,1.0,3.0)
            #x2_gn = trans_gn(x)
            #m_gn = @fsa [partials(x2_gn[1], 1) partials(x2_gn[1], 2) partials(x2_gn[1], 3);
            #             partials(x2_gn[2], 1) partials(x2_gn[2], 2) partials(x2_gn[2], 3);
            #             partials(x2_gn[3], 1) partials(x2_gn[3], 2) partials(x2_gn[3], 3)]
            #m = transform_deriv_params(trans, x)
            #@test m ≈ m_gn

        end
    end

end
