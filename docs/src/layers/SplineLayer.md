# Spline Layer

Constructs a Spline Layer. At a high-level, it performs the following:

 1. Takes as input a one-dimensional training dataset, a time span, a time step and
    an interpolation method.
 2. During training, adjusts the values of the function at multiples of the time-step
    such that the curve interpolated through these points has minimum loss on the corresponding
    one-dimensional dataset.

```@docs
SplineLayer
```
