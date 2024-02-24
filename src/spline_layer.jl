"""
    SplineLayer(time_span, time_step, spline_basis, init_saved_points = nothing)

Constructs a Spline Layer. At a high-level, it performs the following:

 1. Takes as input a one-dimensional training dataset, a time span, a time step and
    an interpolation method.
 2. During training, adjusts the values of the function at multiples of the time-step such
    that the curve interpolated through these points has minimum loss on the corresponding
    one-dimensional dataset.

Arguments:

  - `time_span`: Tuple of real numbers corresponding to the time span.
  - `time_step`: Real number corresponding to the time step.
  - `spline_basis`: Interpolation method to be used yb the basis (current supported
    interpolation methods: `ConstantInterpolation`, `LinearInterpolation`,
    `QuadraticInterpolation`, `QuadraticSpline`, `CubicSpline`).
  - `init_saved_points`: values of the function at multiples of the time step. Initialized by
    default to a random vector sampled from the unit normal. Alternatively, can take a
    function with the signature `init_saved_points(rng, time_span, time_step)`.
"""
@concrete struct SplineLayer <: AbstractExplicitLayer
    tspan
    tstep
    spline_basis
    init_saved_points
end

function SplineLayer(tspan, tstep, spline_basis; init_saved_points::F = nothing) where {F}
    return SplineLayer(tspan, tstep, spline_basis, init_saved_points)
end

function LuxCore.initialparameters(rng::AbstractRNG, l::SplineLayer)
    if l.init_saved_points === nothing
        return (;
            saved_points = randn(rng, typeof(l.tspan[1]),
                length(l.tspan[1]:(l.tstep):l.tspan[2])))
    else
        return (; saved_points = l.init_saved_points(rng, l.tspan, l.tstep))
    end
end

function (layer::SplineLayer)(t, ps, st)
    return (
        layer.spline_basis(ps.saved_points,
            layer.tspan[1]:(layer.tstep):layer.tspan[2])(t),
        st)
end
