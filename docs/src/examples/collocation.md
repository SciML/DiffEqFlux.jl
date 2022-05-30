# Smoothed Collocation for Fast Two-Stage Training

One can avoid a lot of the computational cost of the ODE solver by
pretraining the neural network against a smoothed collocation of the
data. First the example and then an explanation.

```julia
using DiffEqFlux, DifferentialEquations, GalacticOptim, GalacticOptimJL, Plots

u0 = Float32[2.0; 0.0]
datasize = 300
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps)) .+ 0.1randn(2,300)

du,u = collocate_data(data,tsteps,EpanechnikovKernel())

scatter(tsteps,data')
plot!(tsteps,u',lw=5)
savefig("colloc.png")
plot(tsteps,du')
savefig("colloc_du.png")

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))

function loss(p)
    cost = zero(first(p))
    for i in 1:size(du,2)
      _du = dudt2(@view(u[:,i]),p)
      dui = @view du[:,i]
      cost += sum(abs2,dui .- _du)
    end
    sqrt(cost)
end

pinit = initial_params(dudt2)
callback = function (p, l)
  return false
end

result_neuralode = GalacticOptim.solve(loss, pinit,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 10000)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
nn_sol = prob_neuralode(u0, result_neuralode.u)
scatter(tsteps,data')
plot!(nn_sol)
savefig("colloc_trained.png")

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end

@time numerical_neuralode = GalacticOptim.solve(loss_neuralode, result_neuralode.u,
                                                ADAM(0.05), cb = callback,
                                                maxiters = 300)

nn_sol = prob_neuralode(u0, numerical_neuralode.u)
scatter(tsteps,data')
plot!(nn_sol,lw=5)
savefig("post_trained.png")
```

## Generating the Collocation

The smoothed collocation is a spline fit of the datapoints which allows
us to get a an estimate of the approximate noiseless dynamics:

```julia
using DiffEqFlux, DifferentialEquations, Plots

u0 = Float32[2.0; 0.0]
datasize = 300
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps)) .+ 0.1randn(2,300)

du,u = collocate_data(data,tsteps,EpanechnikovKernel())

scatter(tsteps,data')
plot!(tsteps,u',lw=5)
```

![](https://user-images.githubusercontent.com/1814174/87254751-d8177600-c452-11ea-9095-3af303d9b975.png)

We can then differentiate the smoothed function to get estimates of the
derivative at each datapoint:

```julia
plot(tsteps,du')
```

![](https://user-images.githubusercontent.com/1814174/87254752-d8b00c80-c452-11ea-8011-7fd667b87311.png)

Because we have `(u',u)` pairs, we can write a loss function that
calculates the squared difference between `f(u,p,t)` and `u'` at each
point, and find the parameters which minimize this difference:

```julia
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))

function loss(p)
    cost = zero(first(p))
    for i in 1:size(du,2)
      _du = dudt2(@view(u[:,i]),p)
      dui = @view du[:,i]
      cost += sum(abs2,dui .- _du)
    end
    sqrt(cost)
end

pinit = initial_params(dudt2)
callback = function (p, l)
  return false
end

result_neuralode = GalacticOptim.solve(loss, pinit,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 10000)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
nn_sol = prob_neuralode(u0, result_neuralode.u)
scatter(tsteps,data')
plot!(nn_sol)
```

![](https://user-images.githubusercontent.com/1814174/87254749-d8177600-c452-11ea-9643-86c6375fa493.png)

While this doesn't look great, it has the characteristics of the
full solution all throughout the timeseries, but it does have a drift.
We can continue to optimize like this, or we can use this as the
initial condition to the next phase of our fitting:

```julia
function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end

@time numerical_neuralode = GalacticOptim.solve(loss_neuralode, result_neuralode.u,
                                                ADAM(0.05), cb = callback,
                                                maxiters = 300)

nn_sol = prob_neuralode(u0, numerical_neuralode.u)
scatter(tsteps,data')
plot!(nn_sol,lw=5)
```

![](https://user-images.githubusercontent.com/1814174/87254750-d8177600-c452-11ea-8cfa-3f805beaf0aa.png)

This method then has a good global starting position, making it less
prone to local minima and is thus a great method to mix in with other
fitting methods for neural ODEs.
