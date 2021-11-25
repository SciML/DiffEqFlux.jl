"""
`sciml_train`

### Unconstrained Optimization

```julia
function sciml_train(loss, _θ, opt = DEFAULT_OPT, adtype = DEFAULT_AD,
                     _data = DEFAULT_DATA, args...;
                     cb = (args...) -> false, maxiters = get_maxiters(data),
                     kwargs...)
```

### Box Constrained Optimization

```julia
function sciml_train(loss, θ, opt = DEFAULT_OPT, adtype = DEFAULT_AD,
                     data = DEFAULT_DATA, args...;
                     lower_bounds, upper_bounds,
                     cb = (args...) -> (false), maxiters = get_maxiters(data),
                     kwargs...)
```

## Optimizer Choices and Arguments

For a full definition of the allowed optimizers and arguments, please see the
[GalacticOptim.jl](https://galacticoptim.sciml.ai/dev/) documentation. As
sciml_train is an interface over GalacticOptim.jl, all of its optimizers and
arguments can be used from here.

## Loss Functions and Callbacks

Loss functions in `sciml_train` treat the first returned value as the return.
For example, if one returns `(1.0, [2.0])`, then the value the optimizer will
see is `1.0`. The other values are passed to the callback function. The callback
function is `cb(p, args...)` where the arguments are the extra returns from the
loss. This allows for reusing instead of recalculating. The callback function
must return a boolean where if `true`, then the optimizer will prematurely end
the optimization. It is called after every successful step, something that is
defined in an optimizer-dependent manner.

## Default AD Choice

The current default AD choice is dependent on the number of parameters.
For <50 parameters both ForwardDiff.jl and Zygote.jl gradients are evaluated
and the fastest is used. If both methods fail, finite difference method 
is used as a fallback. For ≥50 parameters Zygote.jl is used. 
More refinements to the techniques are planned.

## Default Optimizer Choice

By default, if the loss function is deterministic than an optimizer chain of
ADAM -> BFGS is used, otherwise ADAM is used (and a choice of maxiters is required).
"""
function sciml_train(loss, θ, opt=nothing, adtype=nothing, args...;
                     lower_bounds=nothing, upper_bounds=nothing,
                     maxiters=nothing, kwargs...)
    if adtype === nothing
        if length(θ) < 50
            fdtime = try
                ForwardDiff.gradient(x -> first(loss(x)), θ)
                @elapsed ForwardDiff.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end
            zytime = try
                Zygote.gradient(x -> first(loss(x)), θ)
                @elapsed Zygote.gradient(x -> first(loss(x)), θ)
            catch
                Inf
            end

            if fdtime == zytime == Inf
                @warn "AD methods failed, using numerical differentiation. To debug, try ForwardDiff.gradient(loss, θ) or Zygote.gradient(loss, θ)"
                adtype = GalacticOptim.AutoFiniteDiff()
            elseif fdtime < zytime
                adtype = GalacticOptim.AutoForwardDiff()
            else
                adtype = GalacticOptim.AutoZygote()
            end

        else
            adtype = GalacticOptim.AutoZygote()
        end
    end

    optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
    optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
    optprob = GalacticOptim.OptimizationProblem(optfunc, θ; lb=lower_bounds, ub=upper_bounds, kwargs...)
    if opt !== nothing
        if maxiters !== nothing
            GalacticOptim.solve(optprob, opt, args...; maxiters, kwargs...)
        else
            GalacticOptim.solve(optprob, opt, args...; kwargs...)
        end
    else
        deterministic = first(loss(θ)) == first(loss(θ))
        if (!isempty(args) || !deterministic) && maxiters === nothing
            error("Automatic optimizer determination requires deterministic loss functions (and no data) or maxiters must be specified.")
        end

        if isempty(args) && deterministic
            # If determinsitic then ADAM -> finish with BFGS
            if maxiters === nothing
                res1 = GalacticOptim.solve(optprob, ADAM(0.01), args...; maxiters=300, kwargs...)
            else
                res1 = GalacticOptim.solve(optprob, ADAM(0.01), args...; maxiters, kwargs...)
            end

            optprob2 = GalacticOptim.OptimizationProblem(
                optfunc, res1.u; lb=lower_bounds, ub=upper_bounds, kwargs...)
            res1 = GalacticOptim.solve(
                optprob2, BFGS(initial_stepnorm=0.01), args...; maxiters, kwargs...)
        else
            res1 = GalacticOptim.solve(optprob, ADAM(0.1), args...; maxiters, kwargs...)
        end
    end
end
