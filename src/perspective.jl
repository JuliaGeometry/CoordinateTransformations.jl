"""
    PerspectiveMap()

Construct a perspective transformation. The persepective transformation takes,
e.g., a point in 3D space and "projects" it onto a 2D virtual screen of an ideal
pinhole camera (at distance `1` away from the camera). The camera is oriented
towards the positive-Z axis (or in general, along the final dimension).

This transformation is designed to be used in composition with other coordinate
transformations, defining e.g. the position and orientation of the camera. For
example:

    cam_transform = PerspectiveMap() ∘ AffineMap(cam_rotation, -cam_position)
    screen_points = map(cam_transform, points)
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
