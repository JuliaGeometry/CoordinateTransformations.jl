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

            y = transform(trans, x)
            @test y == R * Vec(1.0, 2.0, 3.0)

            x_gn = Point(GradientNumber(1.0,(1.,0.,0.)), GradientNumber(2.0,(0.,1.,0.)), GradientNumber(3.0,(0.,0.,1.)))
            y_gn = transform(trans, x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[y_gn[i].partials.data[j] for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            g11 = GradientNumber(R[1,1],(1.,0.,0.,0.,0.,0.,0.,0.,0.))
            g12 = GradientNumber(R[1,2],(0.,1.,0.,0.,0.,0.,0.,0.,0.))
            g13 = GradientNumber(R[1,3],(0.,0.,1.,0.,0.,0.,0.,0.,0.))
            g21 = GradientNumber(R[2,1],(0.,0.,0.,1.,0.,0.,0.,0.,0.))
            g22 = GradientNumber(R[2,2],(0.,0.,0.,0.,1.,0.,0.,0.,0.))
            g23 = GradientNumber(R[2,3],(0.,0.,0.,0.,0.,1.,0.,0.,0.))
            g31 = GradientNumber(R[3,1],(0.,0.,0.,0.,0.,0.,1.,0.,0.))
            g32 = GradientNumber(R[3,2],(0.,0.,0.,0.,0.,0.,0.,1.,0.))
            g33 = GradientNumber(R[3,3],(0.,0.,0.,0.,0.,0.,0.,0.,1.))

            G = @fsa [g11 g12 g13;
                      g21 g22 g23;
                      g31 g32 g33 ]

            trans_gn = Rotation(G)
            y_gn = transform(trans_gn, x)

            M_gn = Mat{3,9,Float64}(vcat(ntuple(i->[y_gn[i].partials.data[j] for j = 1:9].', 3)...))
            M = transform_deriv_params(trans, x)
            @test M ≈ M_gn
        end

        @testset "Rotation{Quaternion} (quaternion parameterization)" begin
            v = [0.35, 0.45, 0.25, 0.15]
            v = v / vecnorm(v)
            q = Quaternion(v)

            trans = Rotation(q)
            x = Point(1.0, 2.0, 3.0)

            @test inv(trans) ≈ Rotation(inv(Quaternion(v...)))
            @test trans ∘ trans ≈ Rotation(trans.matrix * trans.matrix)

            y = transform(trans, x)
            @test y ≈ Point(3.439024390243902,-1.1463414634146332,0.9268292682926829)

            x_gn = Point(GradientNumber(1.0,(1.,0.,0.)), GradientNumber(2.0,(0.,1.,0.)), GradientNumber(3.0,(0.,0.,1.)))
            y_gn = transform(trans, x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[y_gn[i].partials.data[j] for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            v_gn = [GradientNumber(v[1],(1.,0.,0.,0.)), GradientNumber(v[2],(0.,1.,0.,0.)), GradientNumber(v[3],(0.,0.,1.,0.)), GradientNumber(v[4],(0.,0.,0.,1.))]
            q_gn = Quaternion(v_gn...)
            trans_gn = Rotation(q_gn)
            y_gn = transform(trans_gn,x)
            M_gn = Mat{3,4,Float64}(vcat(ntuple(i->[y_gn[i].partials.data[j] for j = 1:4].', 3)...))
            M = transform_deriv_params(trans, x)
            @test M ≈ M_gn
        end

        @testset "Rotation{EulerAngles} (Euler angle parameterization)" begin
            θx = 0.1
            θy = 0.2
            θz = 0.3

            trans = Rotation(EulerAngles(θx, θy, θz))
            x = Point(1.0, 2.0, 3.0)

            @test inv(trans) == Rotation(trans.matrix')
            @test trans ∘ trans == Rotation(trans.matrix * trans.matrix)

            y = transform(trans, x)
            @test y == Point(0.9984766744283545,2.1054173473736495,2.92750100324502)

            x_gn = Point(GradientNumber(1.0,(1.,0.,0.)), GradientNumber(2.0,(0.,1.,0.)), GradientNumber(3.0,(0.,0.,1.)))
            y_gn = transform(trans, x_gn)
            M_gn = Mat{3,3,Float64}(vcat(ntuple(i->[y_gn[i].partials.data[j] for j = 1:3].', 3)...))
            M = transform_deriv(trans, x)
            @test M ≈ M_gn

            # Parameter derivative not defined
            # TODO?
        end

        @testset "RotationXY, RotationYZ and RotationZX" begin
            # RotationXY
            x = Point(2.0, 0.0, 0.0)
            x2 = transform(CartesianFromSpherical(), Spherical(2.0, 1.0, 0.0))
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
            @test transform(trans, x) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(GradientNumber(2.0, (1.0,0.0,0.0)), GradientNumber(1.0, (0.0,1.0,0.0)), GradientNumber(3.0, (0.0,0.0,1.0)))
            x2_gn = transform(trans, x_gn)
            m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2] x2_gn[1].partials.data[3];
                         x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] x2_gn[2].partials.data[3];
                         x2_gn[3].partials.data[1] x2_gn[3].partials.data[2] x2_gn[3].partials.data[3] ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationXY(GradientNumber(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = transform(trans_gn, x)
            m_gn = Mat(x2_gn[1].partials.data[1], x2_gn[2].partials.data[1], x2_gn[3].partials.data[1])
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn


            # RotationYZ
            x = Point(0.0, 2.0, 0.0)
            x2 = transform(CartesianFromSpherical(), Spherical(2.0, pi/2, 1.0))
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
            @test transform(trans, x) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(GradientNumber(2.0, (1.0,0.0,0.0)), GradientNumber(1.0, (0.0,1.0,0.0)), GradientNumber(3.0, (0.0,0.0,1.0)))
            x2_gn = transform(trans, x_gn)
            m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2] x2_gn[1].partials.data[3];
                         x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] x2_gn[2].partials.data[3];
                         x2_gn[3].partials.data[1] x2_gn[3].partials.data[2] x2_gn[3].partials.data[3] ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationYZ(GradientNumber(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = transform(trans_gn, x)
            m_gn = Mat(x2_gn[1].partials.data[1], x2_gn[2].partials.data[1], x2_gn[3].partials.data[1])
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn


            # RotationZX
            x = Point(2.0, 0.0, 0.0)
            x2 = transform(CartesianFromSpherical(), Spherical(2.0, 0.0, -1.0))
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
            @test transform(trans, x) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(GradientNumber(2.0, (1.0,0.0,0.0)), GradientNumber(1.0, (0.0,1.0,0.0)), GradientNumber(3.0, (0.0,0.0,1.0)))
            x2_gn = transform(trans, x_gn)
            m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2] x2_gn[1].partials.data[3];
                         x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] x2_gn[2].partials.data[3];
                         x2_gn[3].partials.data[1] x2_gn[3].partials.data[2] x2_gn[3].partials.data[3] ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative
            trans_gn = RotationZX(GradientNumber(1.0, (1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = transform(trans_gn, x)
            m_gn = Mat(x2_gn[1].partials.data[1], x2_gn[2].partials.data[1], x2_gn[3].partials.data[1])
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn
        end

        @testset "euler_rotation() and composed derivatives" begin
            x = Point(2.0,1.0,3.0)
            trans = euler_rotation(0.1,0.2,0.3)
            x2 = Point(2.730537054338937,0.8047190852558106,2.428290466296628)

            @test trans.t1.t1 == RotationXY(0.1)
            @test trans.t1.t2 == RotationYZ(0.2)
            @test trans.t2 == RotationZX(0.3)

            @test inv(trans) == RotationZX(-0.3) ∘ (RotationYZ(-0.2) ∘ RotationXY(-0.1))

            @test transform(trans, x) ≈ x2

            # Transform derivative
            x = Point(2.0,1.0,3.0)
            x_gn = Point(GradientNumber(2.0, (1.0,0.0,0.0)), GradientNumber(1.0, (0.0,1.0,0.0)), GradientNumber(3.0, (0.0,0.0,1.0)))
            x2_gn = transform(trans, x_gn)
            m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2] x2_gn[1].partials.data[3];
                         x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] x2_gn[2].partials.data[3];
                         x2_gn[3].partials.data[1] x2_gn[3].partials.data[2] x2_gn[3].partials.data[3] ]
            m = transform_deriv(trans, x)
            @test m ≈ m_gn

            # Transform parameter derivative

            trans_gn = euler_rotation(GradientNumber(0.1, (1.0, 0.0, 0.0)), GradientNumber(0.2, (0.0, 1.0, 0.0)), GradientNumber(0.3, (0.0, 0.0, 1.0)))
            x = Point(2.0,1.0,3.0)
            x2_gn = transform(trans_gn, x)
            m_gn = @fsa [x2_gn[1].partials.data[1] x2_gn[1].partials.data[2] x2_gn[1].partials.data[3];
                         x2_gn[2].partials.data[1] x2_gn[2].partials.data[2] x2_gn[2].partials.data[3];
                         x2_gn[3].partials.data[1] x2_gn[3].partials.data[2] x2_gn[3].partials.data[3] ]
            m = transform_deriv_params(trans, x)
            @test m ≈ m_gn

        end
    end

end
