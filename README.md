# CoordinateTransformations

[![Build Status](https://github.com/JuliaGeometry/CoordinateTransformations.jl/workflows/CI/badge.svg)](https://github.com/JuliaGeometry/CoordinateTransformations.jl/actions?query=workflow%3ACI)

**CoordinateTransformations** is a Julia package to manage simple or complex
networks of coordinate system transformations. Transformations can be easily
applied, inverted, composed, and differentiated (both with respect to the
input coordinates and with respect to transformation parameters such as rotation
angle). Transformations are designed to be light-weight and efficient enough
for, e.g., real-time graphical applications, while support for both explicit
and automatic differentiation makes it easy to perform optimization and
therefore ideal for computer vision applications such as SLAM (simultaneous
localization and mapping).

The package provide two main pieces of functionality

1. Primarily, an interface for defining `Transformation`s and applying
   (by calling), inverting (`inv()`), composing (`∘` or `compose()`) and
   differentiating (`transform_deriv()` and `transform_deriv_params()`) them.

2. A small set of built-in, composable, primitive transformations for
   transforming 2D and 3D points (optionally leveraging the *StaticArrays*
   and *Rotations* packages).

### Quick start

Let's translate a 3D point:
```julia
using CoordinateTransformations, Rotations, StaticArrays

x = SVector(1.0, 2.0, 3.0)  # SVector is provided by StaticArrays.jl
trans = Translation(3.5, 1.5, 0.0)

y = trans(x)
```

We can either apply different transformations in turn,
```julia
rot = LinearMap(RotX(0.3))  # Rotate 0.3 radians about X-axis, from Rotations.jl

z = trans(rot(x))
```
or build a composed transformation using the `∘` operator (accessible at the
REPL by typing `\circ` then tab):
```julia
composed = trans ∘ rot  # alternatively, use compose(trans, rot)

composed(x) == z
```
A composition of a `Translation` and a `LinearMap` results in an `AffineMap`.

We can invert the transformation:
```julia
composed_inv = inv(composed)

composed_inv(z) == x
```

For any transformation, we can shift the origin to a new point using `recenter`:
```julia
rot_around_x = recenter(rot, x)
```
Now `rot_around_x` is a rotation around the point `x = SVector(1.0, 2.0, 3.0)`.


Finally, we can construct a matrix describing how the components of `z`
differentiates with respect to components of `x`:
```julia
∂z_∂x = transform_deriv(composed, x) # In general, the transform may be non-linear, and thus we require the value of x to compute the derivative
```

Or perhaps we want to know how `y` will change with respect to changes of
to the translation parameters:
```julia
∂y_∂θ = transform_deriv_params(trans, x)
```

### The interface

Transformations are derived from `Transformation`. As an example, we have
`Translation{T} <: Transformation`. A `Translation` will accept and translate
points in a variety of formats, such as `Vector` or `SVector`, but in general
your custom-defined `Transformation`s could transform any Julia object.

Transformations can be reversed using `inv(trans)`. They can be chained
together using the `∘` operator (`trans1 ∘ trans2`) or `compose` function (`compose(trans1, trans2)`).
In this case, `trans2` is applied first to the data, before `trans1`.
Composition may be intelligent, for instance by precomputing a new `Translation`
by summing the elements of two existing `Translation`s, and yet other
transformations may compose to the `IdentityTransformation`. But by default,
composition will result in a `ComposedTransformation` object which simply
dispatches to apply the transformations in the correct order.

Finally, the matrix describing how differentials propagate through a transform
can be calculated with the `transform_deriv(trans, x)` method. The derivatives
of how the output depends on the transformation parameters is accessed via
`transform_deriv_params(trans, x)`. Users currently have to overload these methods,
as no fall-back automatic differentiation is currently included. Alternatively,
all the built-in types and transformations are compatible with automatic differentiation
techniques, and can be parameterized by *DualNumbers*' `DualNumber` or *ForwardDiff*'s `Dual`.

### Built-in transformations

A small number of 2D and 3D coordinate systems and transformations are included.
We also have `IdentityTransform` and `ComposedTransformation`, which allows us
to nest together arbitrary transformations to create a complex yet efficient
transformation chain.

#### Coordinate types

The package accepts any `AbstractVector` type for Cartesian coordinates (as
well as *FixedSizeArrays* types in Julia v0.4 only). For speed, we recommend
using a statically-sized container such as `SVector{N}` from *StaticArrays*.

We do provide a few specialist coordinate types. The `Polar(r, θ)` and `Polard(r, θ)` types are 2D
polar representations of a point (using radians and degrees, respectively). They share the exported abstract type `AbstractPolar`. In 3D we have defined
`Spherical(r, θ, ϕ)` and `Cylindrical(r, θ, z)`.

#### Coordinate system transformations

Two-dimensional coordinates may be converted using these parameterless (singleton)
transformations:

1. `PolarFromCartesian()`
2. `PolardFromCartesian()`
3. `CartesianFromPolar()`

Three-dimensional coordinates may be converted using these parameterless
transformations:

1. `SphericalFromCartesian()`
2. `CartesianFromSpherical()`
3. `SphericalFromCylindrical()`
4. `CylindricalFromSpherical()`
5. `CartesianFromCylindrical()`
6. `CylindricalFromCartesian()`

However, you may find it simpler to use the convenience constructors like
`Polar(SVector(1.0, 2.0))`.

#### Translations

Translations can be be applied to Cartesian coordinates in arbitrary dimensions,
by e.g. `Translation(Δx, Δy)` or `Translation(Δx, Δy, Δz)` in 2D/3D, or by
`Translation(Δv)` in general (with `Δv` an `AbstractVector`). Compositions of
two `Translation`s will intelligently create a new `Translation` by adding the
translation vectors.

#### Linear transformations

Linear transformations (a.k.a. linear maps), including rotations, can be
encapsulated in the `LinearMap` type, which is a simple wrapper of an
`AbstractMatrix`.

You are able to provide any matrix of your choosing, but your choice of type
will have a large effect on speed. For instance, if you know the dimensionality
of your points (e.g. 2D or 3D) you might consider a statically sized matrix
like `SMatrix` from *StaticArrays.jl*. We recommend performing 3D rotations
using those from *Rotations.jl* for their speed and flexibility. Scaling will
be efficient with Julia's built-in `UniformScaling`. Also note that compositions
of two `LinearMap`s will intelligently create a new `LinearMap` by multiplying
the transformation matrices.

#### Affine maps

An Affine map encapsulates a more general set of transformation which are
defined by a composition of a translation and a linear transformation. An
`AffineMap` is constructed from an `AbstractVector` translation `v` and an
`AbstractMatrix` linear transformation `M`. It will perform the mapping
`x -> M*x + v`, but the order of addition and multiplication will be more obvious
(and controllable) if you construct it from a composition of a linear map
and a translation, e.g. `Translation(v) ∘ LinearMap(v)` (or any combination of
`LinearMap`, `Translation` and `AffineMap`).

#### Perspective transformations

The perspective transformation maps real-space coordinates to those on a virtual
"screen" of one lesser dimension. For instance, this process is used to render
3D scenes to 2D images in computer generated graphics and games. It is an ideal
model of how a pinhole camera operates and is a good approximation of the modern
photography process.

The `PerspectiveMap()` command creates a `Transformation` to perform the
projective mapping. It can be applied individually, but is particularly
powerful when composed with an `AffineMap` containing the position and
orientation of the camera in your scene. For example, to transfer `points` in 3D
space to 2D `screen_points` giving their projected locations on a virtual camera
image, you might use the following code:

```julia
cam_transform = PerspectiveMap() ∘ inv(AffineMap(cam_rotation, cam_position))
screen_points = map(cam_transform, points)
```

There is also a `cameramap()` convenience function that can create a composed
transformation that includes the intrinsic scaling (e.g. focal length and pixel
size) and offset (defining which pixel is labeled `(0,0)`) of an imaging system.

## Acknowledgements

[![FugroRoames](https://avatars.githubusercontent.com/FugroRoames?s=150)](https://github.com/FugroRoames)
