# sciml_train

!!! note

    `sciml_train` is planned to be replaced by [GalacticOptim.jl](https://github.com/SciML/GalacticOptim.jl)
    when it is ready. This optimizer library will have a feature superset of `sciml_train`
    but will have a slightly different interface to allow for backprpogation over
    the optimization and handling constrained optimization in a nicer manner. Translation
    from `sciml_train` to GalacticOptim's style will be fairly trivial since the internals
    are largely the same, and deprecation warnings will help you update when the time
    comes, so do not worry about using this functionalty.

`sciml_train` is a multi-package optimization setup. It currently allows for using
the following nonlinear optimization packages as the backend:

- [Flux.jl](https://fluxml.ai/Flux.jl/stable/)
- [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
- [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl)
- [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl)
- [MultistartOptimization.jl](https://github.com/tpapp/MultistartOptimization.jl)
- [Evolutionary.jl](https://github.com/wildart/Evolutionary.jl)

Thus it allows for local and global optimization, both derivative-based and
derivative-free, with first and second order methods, in an easy and optimized
fashion over the scientific machine learning layer functions provided by
DiffEqFlux.jl. These functions come complete with integration with automatic
differentiation to allow for ease of use with first and second order optimization
methods, where Hessians are derived via forward-over-reverse second order
sensitivity analysis.

To use an optimizer from any of these libraries, one must `using` the appropriate
library first.

## API

### Unconstrained Optimization

```julia
function sciml_train(loss, _θ, opt, _data = DEFAULT_DATA;
                     cb = (args...) -> false,
                     maxiters = get_maxiters(data),
                     progress=true, save_best=true)
```

### Box Constrained Optimization

```julia
function sciml_train(loss, θ, opt,
                     data = DEFAULT_DATA;
                     lower_bounds, upper_bounds,
                     cb = (args...) -> (false), maxiters = get_maxiters(data))
```

## Loss Functions and Callbacks

Loss functions in `sciml_train` treat the first returned value as the return.
For example, if one returns `(1.0,[2.0])`, then the value the optimizer will
see is `1.0`. The other values are passed to the callback function. The callback
function is `cb(p,args...)` where the arguments are the extra returns from the
loss. This allows for reusing instead of recalculating. The callback function
must return a boolean where if `true` then the optimizer will prematurely end
the optimization. It is called after every successful step, something that is
defined in an optimizer-dependent manner.
