# Smoothed Collocation

Smoothed collocation, also referred to as the two-stage method, allows
for fitting differential equations to time series data without relying
on a numerical differential equation solver by building a smoothed
collocating polynomial and using this to estimate the true `(u',u)`
pairs, at which point `u'-f(u,p,t)` can be directly estimated as a
loss to determine the correct parameters `p`. This method can be
extremely fast and robust to noise, though, because it does not
accumulate through time, is not as exact as other methods.

!!! note
    
    This is one of many methods for calculating the collocation coefficients
    for the training process. For a more comprehensive set of collocation
    methods, see [JuliaSimModelOptimizer](https://help.juliahub.com/jsmo/stable/manual/collocation/).

```@docs
collocate_data
```

## Kernel Choice

Note that the kernel choices of DataInterpolations.jl, such as `CubicSpline()`,
are exact, i.e. go through the data points, while the smoothed kernels are
regression splines. Thus `CubicSpline()` is preferred if the data is not too
noisy or is relatively sparse. If data is sparse and very noisy, a `BSpline()`
can be the best regression spline, otherwise one of the other kernels such as as
`EpanechnikovKernel`.

## Non-Allocating Forward-Mode L2 Collocation Loss

The following is an example of a loss function over the collocation that
is non-allocating and compatible with forward-mode automatic differentiation:

```julia
using PreallocationTools
du = PreallocationTools.dualcache(similar(prob.u0))
preview_est_sol = [@view estimated_solution[:, i] for i in 1:size(estimated_solution, 2)]
preview_est_deriv = [@view estimated_derivative[:, i]
                     for i in 1:size(estimated_solution, 2)]

function construct_iip_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    function (p)
        _du = PreallocationTools.get_tmp(du, p)
        vecdu = vec(_du)
        cost = zero(first(p))
        for i in 1:length(preview_est_sol)
            est_sol = preview_est_sol[i]
            f(_du, est_sol, p, tpoints[i])
            vecdu .= vec(preview_est_deriv[i]) .- vec(_du)
            cost += sum(abs2, vecdu)
        end
        sqrt(cost)
    end
end
cost_function = construct_iip_cost_function(
    f, du, preview_est_sol, preview_est_deriv, tpoints)
```
