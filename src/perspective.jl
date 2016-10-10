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
immutable PerspectiveMap <: Transformation
end

function (::PerspectiveMap)(v::AbstractVector)
    scale =  1/v[end]
    return [v[i] * scale for i in 1:length(v)-1]
end

@inline function (::PerspectiveMap)(v::StaticVector)
    return pop(v) * inv(v[end])
end

Base.@pure Base.isapprox(::PerspectiveMap, ::PerspectiveMap; kwargs...) = true

"""
    cameramap(property = value, ...)

Create a transformation that takes points in real space (e.g. 3D) and projects
them through a perspective transformation onto the focal plane of an ideal
(pinhole) camera with the given properties.

All properties are optional. Valid properties include:

 * `focal_length` (in physical units)
 * `pixel_size` (in physical units) or `pixel_size_x` and `pixel_size_y`
 * `offset_x` and `offset_y` (in pixels)
 * `origin` (a vector) and `orientation` (a rotation matrix)

By default, the camera looks towards the postive-`z` axis from `(0,0,0)` and
the sign of the `x` and `y` components is preserved for objects in front of the
camera (objects behind the camera are also projected and therefor inverted - it
is up to the user to cull these as necessary).

If `origin` and `orientation` are specified, the camera is translated to `origin`
and rotated by `orientation` before the perspective map is applied.

(see also `PerspectiveMap`)
"""
function cameramap(;focal_length = nothing,
                    pixel_size = nothing,
                    pixel_size_x = nothing,
                    pixel_size_y = nothing,
                    offset_x = nothing,
                    offset_y = nothing,
                    origin = nothing,
                    orientation = nothing)

    trans = PerspectiveMap()

    if pixel_size === nothing # (this form of if-else-end is handled well by the compiler... the author is looking forward to v0.6 where !(::Bool) is pure...)
    else
        pixel_size_x = pixel_size
        pixel_size_y = pixel_size
    end

    # Apply camera rotations, if necessary
    if origin === nothing && orientation === nothing
    else
        trans = trans ∘ inv(AffineMap(orientation, origin))
    end

    # Apply camera scaling, if necessary
    if isa(focal_length, Void) && isa(pixel_size_x, Void) && isa(pixel_size_y, Void)
    else
        trans = LinearMap(UniformScaling(focal_length/pixel_size)) ∘ trans
    end

    # Apply pixel offset, if necessary
    if isa(offset_x, Void) && isa(offset_y, Void)
    else
        trans = Translation(SVector(-offset_x, -offset_y)) ∘ trans
    end

    return trans
end
