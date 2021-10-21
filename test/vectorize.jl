#
N = 1000
xy  = rand(2,N)
xyz = rand(3,N)

v2 = rand(2)
v3 = rand(3)
tr2 = Translation(v2)
tr3 = Translation(v3)

M2 = rand(2,2)
M3 = rand(3,3)
lr2 = LinearMap(M2)
lr3 = LinearMap(M3)
tc2 = CoordinateTransformations.TransformColumns(tr2)
tc3 = CoordinateTransformations.TransformColumns(tr3)

@test tr2(xy)  == v2 .+ xy
@test tr3(xyz) == v3 .+ xyz
@test lr2(xy)  == M2  * xy
@test lr3(xyz) == M3  * xyz
@test tc2(xy)  == v2 .+ xy

@btime tr2(xy)
@btime tc2(xy)
@btime tr3(xyz)
@btime tc3(xyz)

@test transform_deriv(tr2,xy[:,1]) == LinearAlgebra.I
@test transform_deriv(tr3,xyz)     == LinearAlgebra.I
@test transform_deriv(lr2,xy)      == M2
@test transform_deriv(lr3,xyz)     == M3


