"""
    train!(loss, params, data, opt; cb)
For each datapoint `d` in `data` computes the gradient of `loss(p,d...)` through
backpropagation and calls the optimizer `opt`.
Takes a callback as keyword argument `cb`. For example, this will print "training"
every 10 seconds:
```julia
Flux.train!(loss, params, data, opt,
            cb = throttle(() -> println("training"), 10))
```
The callback can call `Flux.stop()` to interrupt the training loop.
Multiple optimisers and callbacks can be passed to `opt` and `cb` as arrays.
"""
function train!(loss, θ, data, opt; cb = () -> ())
  # Flux is silly and doesn't have an abstract type on its optimizers, so assume
  # this is a Flux optimizer
  cb = Flux.runall(cb)
  @progress for d in data
    try
      local x
      gs = Zygote.gradient(θ) do
        x = loss(θ,d...)
        first(x)
      end
      Flux.update!(opt, ps, gs)
      cb(θ,Base.tail(x)...)
    catch ex
      if ex isa StopException
        break
      else
        rethrow(ex)
      end
    end
  end
end

decompose_trace(trace::Optim.OptimizationTrace) = last(trace)
decompose_trace(trace) = trace

function train!(loss, θ, data, opt::Optim.AbstractOptimizer; cb = () -> ())
  x_tail = nothing
  _cb(trace) = cb(decompose_trace(trace),x_tail...)
  function optim_loss(θ)
    x = loss(θ,next(data))
    x_tail = Base.tail(x)
    first(x)
  end

  function optim_loss_gradient!(g,θ)
    g .= Zygote.gradient(loss_adjoint,θ)[1]
  end
  result =  optimize(optim_loss, optim_loss_gradient!, θ, opt, Optim.Options(extended_trace=true,callback = cb))
end
