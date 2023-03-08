abstract type AbstractSplineLayer <: Function end

"""
Constructs a Spline Layer. At a high-level, it performs the following:
1. Takes as input a one-dimensional training dataset, a time span, a time step and
an interpolation method.
2. During training, adjusts the values of the function at multiples of the time-step
such that the curve interpolated through these points has minimum loss on the corresponding
one-dimensional dataset.

```julia
SplineLayer(time_span,time_step,spline_basis,saved_points=nothing)
```
Arguments:
- `time_span`: Tuple of real numbers corresponding to the time span.
- `time_step`: Real number corresponding to the time step.
- `spline_basis`: Interpolation method to be used yb the basis (current supported
  interpolation methods: ConstantInterpolation, LinearInterpolation, QuadraticInterpolation,
  QuadraticSpline, CubicSpline).
- 'saved_points': values of the function at multiples of the time step. Initialized by default
to a random vector sampled from the unit normal.
"""
struct SplineLayer{T<:Tuple{Real, Real},R<:Real,S1<:AbstractVector,S2<:UnionAll} <: AbstractSplineLayer
    time_span::T
    time_step::R
    saved_points::S1
    spline_basis::S2
    function SplineLayer(time_span,time_step,spline_basis,saved_points=nothing)
        saved_points = randn(length(time_span[1]:time_step:time_span[2]))
        new{typeof(time_span),typeof(time_step),typeof(saved_points),typeof(spline_basis)}(time_span,time_step,saved_points,spline_basis)
    end
end

@functor SplineLayer (saved_points,)

function (layer::SplineLayer)(t::Real,p=layer.saved_points)
    return layer.spline_basis(p,layer.time_span[1]:layer.time_step:layer.time_span[2])(t)
end
