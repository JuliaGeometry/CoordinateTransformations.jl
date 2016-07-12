# CoordinateTransformations

[![Build Status](https://travis-ci.org/FugroRoames/CoordinateTransformations.jl.svg?branch=master)](https://travis-ci.org/FugroRoames/CoordinateTransformations.jl)

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
   transforming 2D and 3D points (optionally leveraging the *FixedSizeArrays*
   and *Rotations* packages).

### Quick start

Let's rotate a 2D point:
```julia
x = Point(1.0, 2.0) # Point is provided by FixedSizeArrays
rot = Rotation2D(0.3) # a rotation by 0.3 radians anticlockwise about the origin

y = rot(x)
```

We can either apply transformations in turn,
```julia
trans = Translation(3.5, 1.5)

z = trans(rot(x))
```
or build a composed transformation:
```julia
composed = trans ∘ rot # or compose(trans, rot)

composed(x) == z
```

We can invert it:
```julia
composed_inv = inv(composed)

composed_inv(z) == x
```

Finally, we can construct a matrix describing how the components of `z`
differentiates with respect to components of `x`:
```julia
∂z_∂x = transform_deriv(composed, x) # In general, the transform may be non-linear, and thus we require the value of x to compute the derivative
```

Or perhaps we want to know how `y` will change with respect to changes of
rotation angle:
```julia
∂y_∂θ = transform_deriv_params(rot, x)
```

### The interface

Transformations are derived from `Transformation`. As an example, we have
`Translation{T} <: Transformation`. A translation will accept points in a
variety of formats, such as `Vector`, `FixedVector`, `Tuple`, etc, and will try
to return the same type as given (of course, type promotion may occur for the
element type itself).

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

The package does not assume any specific coordinate types for Cartesian
coordinates, and aims to accept any indexable container (such as `Vector`,
`Tuple`, *FixedSizeArrays*' `FixedSizeVector{N}` or any other duck-typed vector).
For speed, we recommend using a statically-sized container such as `Point{N}` or
`Vec{N}` from *FixedSizeArrays*,  or even an `NTuple{N}`. However, it is
attempted that the package will not change your data type.

We do provide a few specialist coordinate types. The `Polar(r, θ)` type is a 2D
polar representation of a point, and similarly in 3D we have defined
`Spherical(r, θ, ϕ)` and `Cylindrical(r, θ, z)`.

#### Coordinate system transformations

Two-dimensional coordinates may be converted using these parameterless (singleton)
transformations:

1. `PolarFromCartesian()`
2. `CartesianFromPolar()`

Three-dimensional coordinates may be converted using these parameterless
transformations:

1. `SphericalFromCartesian()`
2. `CartesianFromSpherical()`
3. `SphericalFromCylindrical()`
4. `CylindricalFromSpherical()`
5. `CartesianFromCylindrical()`
6. `CylindricalFromCartesian()`

#### Translations

Translations can be be applied to Cartesian coordinates in arbitrary dimensions,
by e.g. `Translation(Δx, Δy)` or `Translation(Δx, Δy, Δz)` in 2D/3D.

#### Rotations

Rotations in 2D and 3D are treated slightly differently. The `Rotation2D(Δθ)` rotates a
Cartesian point about the origin, while `RotationPolar(Δθ)` rotates a `Polar`
coordinate about the origin.

`Rotation` performs rotations of 3D Cartesian coordinates. Rotations may be
defined simply by their transformation matrix (assumed to be orthogonal),
or via any of the types exported from the *Rotations* package, including
`Quaternion` (originating from the *Quaternions* package), `AngleAxis`,
`EulerAngles`, `ProperEulerAngles` and `SpQuat`. In these latter cases, the
"parameters" differentiated by `transform_deriv_params()` come directly from the
parameterization of the quaternion, Euler angles, etc. **Note:** the derivative
is only guaranteed to be correct in the tangent plane of the constrained variable,
such as normalized quaternions, and optimization techniques should take this
into account as necessary.

Also included are built-in `RotationXY`, `RotationYZ` and `RotationZX` for
rotating about the *z*, *x* and *y* axes, respectively. A helper function for
constructing a composed transformation of Euler angles is
`euler_rotation(θ₁, θ₂, θ₃, [order = EulerZXY])`, where `order` can be any of
*Rotations*' orderings (`EulerXYZ`, etc). The default `EulerZXY` first rotates
around the *y* axis (*z-x* plane) by `θ₃`, then the *x* axis (*y-z* plane) by
`θ₂`, and finally about the *z* axis (*x-y* plane) by `θ₁`, and is therefore
equivalent to `RotationXY(θ₁) ∘ RotationYZ(θ₂) ∘ RotationZX(θ₃)`.
