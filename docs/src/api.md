# API Reference

## Transformations
```@docs
Transformation
CoordinateTransformations.ComposedTransformation
IdentityTransformation
PerspectiveMap
inv
cameramap
compose
recenter
transform_deriv
transform_deriv_params
```

## Affine maps
```@docs
AbstractAffineMap
AffineMap
AffineMap(::Transformation, ::Any)
AffineMap(::Pair)
LinearMap
Translation
```

## 2D Coordinates
```@docs
Polar
PolarFromCartesian
CartesianFromPolar
```

## 3D Coordinates
```@docs
Cylindrical
Spherical
```

```@docs
CartesianFromCylindrical
CartesianFromSpherical
CylindricalFromCartesian
CylindricalFromSpherical
SphericalFromCartesian
SphericalFromCylindrical
```
