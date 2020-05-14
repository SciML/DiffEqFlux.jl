# Strategies to Avoid Local Minima

Local minima can be an issue with fitting neural differential equations. However,
there are many strategies to avoid local minima:

1. Insert stochasticity into the loss function through minibatching
2. Weigh the loss function to allow for fitting earlier portions first
3. Iteratively grow the fit

In this example we will show how to use strategy (3) in order to increase the
robustness of the fit. Let's start with the same neural ODE example we've used
before except with one small twist: we wish to find the neural ODE that fits
on `(0,5.0)`. Naively, we use the same training strategy as before:

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 5.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 16, tanh),
                  FastDense(16, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Vern7(), saveat = tsteps, abstol=1e-6, reltol=1e-6)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, (ode_data[:,1:size(pred,2)] .- pred))
    return loss, pred
end

iter = 0
callback = function (p, l, pred; doplot = true)
  global iter
  iter += 1

  display(l)
  if doplot
    # plot current prediction against data
    plt = scatter(tsteps[1:size(pred,2)], ode_data[1,1:size(pred,2)], label = "data")
    scatter!(plt, tsteps[1:size(pred,2)], pred[1,:], label = "prediction")
    display(plot(plt))
  end

  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 300)

callback(result_neuralode2.minimizer,loss_neuralode(result_neuralode.minimizer)...;doplot=true)
savefig("local_minima.png")
```

![](https://user-images.githubusercontent.com/1814174/81901710-f82ed400-958c-11ea-993f-118f5513d170.png)

However, we've now fallen into a trap of a local minima. If the optimizer changes
the parameters so it dips early, it will increase the loss because there will
be more error in the later parts of the time series. Thus it tends to just stay
flat and never fit perfectly. This thus suggests strategies (2) and (3): do not
allow the later parts of the time series to influence the fit until the later
stages. Strategy (3) seems to be more robust, so this is what will be demonstrated.

Let's start by reducing the timespan to `(0,1.5)`:

```julia
prob_neuralode = NeuralODE(dudt2, (0.0,1.5), Tsit5(), saveat = tsteps[tsteps .<= 1.5])

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                           ADAM(0.05), cb = callback,
                                           maxiters = 300)

callback(result_neuralode2.minimizer,loss_neuralode(result_neuralode2.minimizer)...;doplot=true)
savefig("shortplot1.png")
```

![](https://user-images.githubusercontent.com/1814174/81901707-f82ed400-958c-11ea-9e8e-0efb10d9b05c.png)

This fits beautifully. Now let's grow the timespan and utilize the parameters
from our `(0,1.5)` fit as the initial condition to our next fit:

```julia
prob_neuralode = NeuralODE(dudt2, (0.0,3.0), Tsit5(), saveat = tsteps[tsteps .<= 3.0])

result_neuralode3 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode2.minimizer,
                                           ADAM(0.05), maxiters = 300,
                                           cb = callback)
callback(result_neuralode3.minimizer,loss_neuralode(result_neuralode3.minimizer)...;doplot=true)
savefig("shortplot2.png")
```

![](https://user-images.githubusercontent.com/1814174/81901706-f7963d80-958c-11ea-856a-7f85af8695b8.png)

Once again a great fit. Now we utilize these parameters as the initial condition
to the full fit:

```julia
prob_neuralode = NeuralODE(dudt2, (0.0,5.0), Tsit5(), saveat = tsteps)

result_neuralode4 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode3.minimizer,
                                           ADAM(0.01), maxiters = 300,
                                           cb = callback)
callback(result_neuralode4.minimizer,loss_neuralode(result_neuralode4.minimizer)...;doplot=true)
savefig("fullplot.png")
```

![](https://user-images.githubusercontent.com/1814174/81901711-f82ed400-958c-11ea-9ba2-2b1f213b865a.png)

And there we go, a robust strategy for fitting an equation that would otherwise
get stuck in a local optima.
