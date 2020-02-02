"""
    train!(loss, params, data, opt; cb)
For each datapoint `d` in `data` computes the gradient of `loss(p,d...)` through
backpropagation and calls the optimizer `opt`.
Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:
```julia
DiffEqFlux.sciml_train!(loss, params, data, opt,
            cb = throttle(() -> println("training"), 10))
```
The callback can call `Flux.stop()` to interrupt the training loop.
Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function sciml_train!(loss, _θ, data, opt; cb = (args...) -> ())
  θ = copy(_θ)
  ps = Flux.params(θ)
  # Flux is silly and doesn't have an abstract type on its optimizers, so assume
  # this is a Flux optimizer
  @progress for d in data
    try
      local x
      gs = Flux.Zygote.gradient(ps) do
        x = loss(θ)
        first(x)
      end
      Flux.Optimise.update!(opt, ps, gs)
      cb(θ,x...)
    catch ex
      if ex isa Flux.Optimise.StopException
        break
      else
        rethrow(ex)
      end
    end
  end
  θ
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function sciml_train!(loss, θ, data, opt::Optim.AbstractOptimizer; cb = (args...) -> ())
  local x
  _cb(trace) = (cb(decompose_trace(trace).metadata["x"],x...);false)
  function optim_loss(θ)
    x = loss(θ)
    first(x)
  end

  function optim_loss_gradient!(g,θ)
    g .= Zygote.gradient(optim_loss,θ)[1]
    nothing
  end
  result =  optimize(optim_loss, optim_loss_gradient!, θ, opt, Optim.Options(extended_trace=true,callback = _cb))
  result.minimizer
end
