"""
    train!(loss, params, data, opt; cb)
For each datapoint `d` in `data` computes the gradient of `loss(p,d...)` through
backpropagation and calls the optimizer `opt`.
Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:
```julia
DiffEqFlux.sciml_train(loss, params, data, opt,
            cb = throttle(() -> println("training"), 10))
```
The callback can call `Flux.stop()` to interrupt the training loop.
Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function sciml_train(loss, _θ, opt; cb = (args...) -> (false), maxiters)
  θ = copy(_θ)
  ps = Flux.params(θ)
  data = Iterators.repeated((), maxiters)
  t0 = time()
  # Flux is silly and doesn't have an abstract type on its optimizers, so assume
  # this is a Flux optimizer
  local x
  @progress for d in data
    gs = Flux.Zygote.gradient(ps) do
      x = loss(θ)
      first(x)
    end
    Flux.Optimise.update!(opt, ps, gs)
    cb_call = cb(θ,x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    elseif cb_call
      break
    end
  end
  _time = time()
  Optim.MultivariateOptimizationResults(opt,
                                        _θ,# initial_x,
                                        θ, #pick_best_x(f_incr_pick, state),
                                        first(x), # pick_best_f(f_incr_pick, state, d),
                                        maxiters, #iteration,
                                        maxiters >= maxiters, #iteration == options.iterations,
                                        false, # x_converged,
                                        0.0,#T(options.x_tol),
                                        0.0,#T(options.x_tol),
                                        NaN,# x_abschange(state),
                                        NaN,# x_abschange(state),
                                        false,# f_converged,
                                        0.0,#T(options.f_tol),
                                        0.0,#T(options.f_tol),
                                        NaN,#f_abschange(d, state),
                                        NaN,#f_abschange(d, state),
                                        false,#g_converged,
                                        0.0,#T(options.g_tol),
                                        NaN,#g_residual(d),
                                        false, #f_increased,
                                        nothing,
                                        maxiters,
                                        maxiters,
                                        0,
                                        true,
                                        NaN,
                                        _time-t0,)
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function sciml_train(loss, θ, opt::Optim.AbstractOptimizer;
                      cb = (args...) -> (false), maxiters = 0)
  local x
  function _cb(trace)
    cb_call = cb(decompose_trace(trace).metadata["x"],x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    end
    cb_call
  end

  function optim_loss(θ)
    x = loss(θ)
    first(x)
  end

  function optim_loss_gradient!(g,θ)
    g .= Flux.Zygote.gradient(optim_loss,θ)[1]
    nothing
  end
  optimize(optim_loss, optim_loss_gradient!, θ, opt,
           Optim.Options(extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters))
end

function sciml_train(loss, lower_bounds, upper_bounds, θ, opt::Optim.AbstractConstrainedOptimizer;
                      cb = (args...) -> (false), maxiters = 0)
  local x
  function _cb(trace)
    cb_call = cb(decompose_trace(trace).metadata["x"],x...)
    if !(typeof(cb_call) <: Bool)
      error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
    end
    cb_call
  end
  function optim_loss(θ)
    x = loss(θ)
    first(x)
  end
  function optim_loss_gradient!(g,θ)
    g .= Flux.Zygote.gradient(optim_loss,θ)[1]
    nothing
  end
  optimize(optim_loss, optim_loss_gradient!, lower_bounds, upper_bounds, θ, opt,
           Optim.Options(extended_trace=true,callback = _cb,
                         f_calls_limit = maxiters))
end
