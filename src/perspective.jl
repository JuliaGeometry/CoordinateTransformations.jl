"""
    PerspectiveMap()

Construct a perspective transformation. The persepective transformation takes,
e.g., a point in 3D space and "projects" it onto a 2D virtual screen of an ideal
pinhole camera (at distance `1` away from the camera). The camera is oriented
towards the positive-Z axis (or in general, along the final dimension) and the
sign of the `x` and `y` components is preserved for objects in front of the
camera (objects behind the camera are also projected and therefore inverted - it
is up to the user to cull these as necessary).

This transformation is designed to be used in composition with other coordinate
transformations, defining e.g. the position and orientation of the camera. For
example:

    cam_transform = PerspectiveMap() ∘ inv(AffineMap(cam_rotation, cam_position))
    screen_points = map(cam_transform, points)

(see also `cameramap`)
"""
struct PerspectiveMap <: Transformation
end

function (::PerspectiveMap)(v::AbstractVector)
    scale =  1/v[end]
    return [v[i] * scale for i in 1:length(v)-1]
end

@inline function (::PerspectiveMap)(v::StaticVector)
    return pop(v) * inv(v[end])
end

Base.isapprox(::PerspectiveMap, ::PerspectiveMap; kwargs...) = true

"""
    cameramap()
    cameramap(scale)
    cameramap(scale, offset)

Create a transformation that takes points in real space (e.g. 3D) and projects
them through a perspective transformation onto the focal plane of an ideal
(pinhole) camera with the given properties.

The `scale` sets the scale of the screen. For a standard digital camera, this
would be `scale = focal_length / pixel_size`. Non-square pixels are supported
by providing a pair of scales in a tuple, `scale = (scale_x, scale_y)`. Positive
scales represent a camera looking in the +z axis with a virtual screen in front
of the camera (the x,y coordinates are not inverted compared to 3D space). Note
that points behind the camera (with negative z component) will be projected
(and inverted) onto the image coordinates and it is up to the user to cull
such points as necessary.

The `offset = (offset_x, offset_y)` is used to define the origin in the imaging
plane. For instance, you may wish to have the point (0,0) represent the top-left
corner of your imaging sensor. This measurement is in the units after applying
`scale` (e.g. pixels).

(see also `PerspectiveMap`)
"""
cameramap() = PerspectiveMap()
cameramap(scale::Number) =
    LinearMap(UniformScaling(scale)) ∘ PerspectiveMap()
cameramap(scale::Tuple{Number, Number}) =
    LinearMap(@SMatrix([scale[1] 0; 0 scale[2]])) ∘ PerspectiveMap()
cameramap(scale::Number, offset::Tuple{Number,Number}) =
    AffineMap(UniformScaling(scale), SVector(-offset[1], -offset[2])) ∘ PerspectiveMap()
cameramap(scale::Tuple{Number, Number}, offset::Tuple{Number,Number}) =
    AffineMap(@SMatrix([scale[1] 0; 0 scale[2]]), SVector(-offset[1], -offset[2])) ∘ PerspectiveMap()

