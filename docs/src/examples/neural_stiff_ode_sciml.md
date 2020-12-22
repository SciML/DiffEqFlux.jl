# Use sciml_train to train a neural network on a stiff chemical reaction system

This tutorial goes into the training of stiff ODEs. Please read the [neural ordinary differential equations tutorial](https://diffeqflux.sciml.ai/dev/examples/neural_ode_sciml/) first.

## Copy-Pasteable Code

Before getting to the explanation, here's some code to start with. We will
follow a full explanation of the definition and training process:

```julia
using Flux, DiffEqFlux, OrdinaryDiffEq, Optim, Plots, Random, DiffEqSensitivity

using ForwardDiff

function f!(du,u,p,t)
  y₁,y₃ = u
  k₁,k₂,k₃ = p
  y₂ = 1 - (y₁+y₃)
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₂*y₂^2
  nothing
end

u₀ = [1.0, 0]

tspan = (0.0, 1e5)
p = [0.04, 3e7, 1e4]
datasize = 100
t = range(tspan[1], tspan[2], length = datasize)

stiff_func = ODEFunction(f!)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(), saveat = t)
sol_stiff_data = Array(sol_stiff)

nn_dudt2 = FastChain(FastDense(2, 4, tanh, initW = (m, n) -> 1e-5rand(MersenneTwister(1), m, n)),
                     FastDense(4, 2, initW = (m, n) -> 1e-5rand(MersenneTwister(3), m, n)))

model_stiff_node = NeuralODE(nn_dudt2,
                             tspan, Tsit5(), saveat = sol_stiff.t,
                             reltol = 1e-3,
                             abstol = 1e-6,
                             maxiters = 1000,
                             sensealg = QuadratureAdjoint(),
                             verbose=false) # Tsit5 is good enough here
model_stiff_node(u₀)

function predict_stiff_node(p)
    return model_stiff_node(u₀, p)
end

function loss_stiff_node(p)
    prediction = predict_stiff_node(p)
    loss = sqrt(sum(abs2, sol_stiff_data .- prediction))
    return loss, prediction
end

callback = function (p, l, pred) #callback function to observe training
  display(l)
  return false
end

l1 = first(loss_stiff_node(model_stiff_node.p))
pp = model_stiff_node.p
for _ in 1:5
  global pp
  result_lbfgs = DiffEqFlux.sciml_train(loss_stiff_node, pp,
                                        LBFGS(),
                                        cb = callback, maxiters = 100)
  pp = result_lbfgs.minimizer
end
plot(plot(sol_stiff, lab=false, title="Ground truth"), plot(predict_stiff_node(pp), lab=false, title="Prediction"))
```
![rober fitting]((https://user-images.githubusercontent.com/17304743/102832372-28a8dd00-43bc-11eb-93de-36166e8b17cf.png)

## Explanation

First, let's get a time series array from the Robertson's equation as data. Note
that since `y₂` has the fastest rate, we want to eliminate it from our training.
We know the conservation law `y₁(t) + y₂(t) + y₃(t) = 1` for all `t`. Hence, we
can always compute `y₂` from `y₁` and `y₃`.

```julia
using Flux, DiffEqFlux, OrdinaryDiffEq, Optim, Plots, Random, DiffEqSensitivity

using ForwardDiff

function f!(du,u,p,t)
  y₁,y₃ = u
  k₁,k₂,k₃ = p
  y₂ = 1 - (y₁+y₃)
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₂*y₂^2
  nothing
end

u₀ = [1.0, 0]

tspan = (0.0, 1e5)
p = [0.04, 3e7, 1e4]
datasize = 100
t = range(tspan[1], tspan[2], length = datasize)

stiff_func = ODEFunction(f!)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(), saveat = t)
sol_stiff_data = Array(sol_stiff)
```

Now let's define a neural network with a `NeuralODE` layer. First we define
the layer. Here we're going to use `FastChain`, which is a faster neural network
structure for NeuralODEs. Since we reduced the fastest rate, the system is not
very stiff anymore. We can use `Tsit5()` non-stiff solver to fit this model for
better efficiency.

```julia
nn_dudt2 = FastChain(FastDense(2, 4, tanh, initW = (m, n) -> 1e-5rand(MersenneTwister(1), m, n)),
                     FastDense(4, 2, initW = (m, n) -> 1e-5rand(MersenneTwister(3), m, n)))

model_stiff_node = NeuralODE(nn_dudt2,
                             tspan, Tsit5(), saveat = sol_stiff.t,
                             reltol = 1e-3,
                             abstol = 1e-6,
                             maxiters = 1000,
                             sensealg = QuadratureAdjoint(),
                             verbose=false) # Tsit5 is good enough here
```

From here we build a loss function around it. The `NeuralODE` has an optional
second argument for new parameters which we will use to iteratively change the
neural network in our training loop. We will use the L2 loss of the network's
output against the time series data:

```julia
function predict_stiff_node(p)
    return model_stiff_node(u₀, p)
end

function loss_stiff_node(p)
    prediction = predict_stiff_node(p)
    loss = sqrt(sum(abs2, sol_stiff_data .- prediction))
    return loss, prediction
end
```

We define a callback function.
```julia
callback = function (p, l, pred) #callback function to observe training
  display(l)
  return false
end
```

We then train the neural network to learn the ODE. Here, we use the `LBFGS`
optimizer, because it yields the best result empirically.
```julia
pp = model_stiff_node.p
for _ in 1:5
  global pp
  result_lbfgs = DiffEqFlux.sciml_train(loss_stiff_node, pp,
                                        LBFGS(),
                                        cb = callback, maxiters = 100)
  pp = result_lbfgs.minimizer
end
```

Finally, we can plot the result.
```julia
plot(plot(sol_stiff, lab=false, title="Ground truth"), plot(predict_stiff_node(pp), lab=false, title="Prediction"))
```
