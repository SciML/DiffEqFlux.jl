"""
```julia
solution = sciml_train(loss, θ, opt [, adtype])
```

Trains a scientific machine learning problem to minimize the specified  `loss` function with 
parameters θ. The optimization backend is specified by `opt`, and optionallly the type of automatic
differentiation `adtype`. Available backends include Flux, Optim, and others, e.g. 
`BFGS()`.

The returned `solution` is an `Optim.MultivariateOptimizationResults` struct, with 
notable fields `minimum` loss value, number of `iterations`, `time_run` in seconds,
`stopped_by` criteria as symbols, and optimization algorithm `method` with parameters. 
Boolean fields describe whether `iteration_converged`, `x_converged`, and `is_success`.

Optional input arguments `args` and keyword arguments `kwargs` are splatted to the
optimization problem and solve functions. Notable kwarg specifies the `cb = callback`
function, which is called after each step, with input arguments θ and the `loss` value.

"""
function sciml_train(loss, θ, opt, adtype::DiffEqBase.AbstractADType = GalacticOptim.AutoZygote(), args...; kwargs...)
  optf = GalacticOptim.OptimizationFunction((x, p) -> loss(x), adtype)
  optfunc = GalacticOptim.instantiate_function(optf, θ, adtype, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, θ; kwargs...)
  GalacticOptim.solve(optprob, opt, args...; kwargs...)
end
