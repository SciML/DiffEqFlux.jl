# Neural Ordinary Differential Equations

Next we define a single layer neural network that using the [AD-compatible
`concrete_solve`
function](https://docs.juliadiffeq.org/latest/analysis/sensitivity/) function
that takes the parameters and an initial condition and returns the solution of
the differential equation as a
[`DiffEqArray`](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl) (same
array semantics as the standard differential equation solution object but
without the interpolations).

We can use DiffEqFlux.jl to define, solve, and train neural ordinary
differential equations. A neural ODE is an ODE where a neural network defines
its derivative function. Thus for example, with the multilayer perceptron neural
network `Chain(Dense(2, 50, tanh), Dense(50, 2))`, the best way to define a
neural ODE by hand would be to use non-mutating adjoints, which looks like:

```
model = Chain(Dense(2, 50, tanh), Dense(50, 2))
p, re = Flux.destructure(model)
dudt!(u, p, t) = re(p)(u)
u0 = rand(2)
prob = ODEProblem(dudt!, u0, tspan, p)
my_neural_ode_prob = concrete_solve(prob, Tsit5(), u0, p, args...; kwargs...)
nothing # hide
```

(`Flux.restructure` and `Flux.destructure` are helper functions which transform
the neural network to use parameters `p`)

A convenience function which handles all of the details is `NeuralODE`. To use
`NeuralODE`, you give it the initial condition, the internal neural network
model to use, the timespan to solve on, and any ODE solver arguments. For
example, this neural ODE would be defined as:

```
tspan = (0.0f0, 25.0f0)
n_ode = NeuralODE(model, tspan, Tsit5(), saveat = 0.1)
nothing # hide
```

where here we made it a layer that takes in the initial condition and spits out
an array for the time series saved at every 0.1 time steps.

### Training a Neural Ordinary Differential Equation

Let's get a time series array from the Lotka-Volterra equation as data:

```@example nnode
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
nothing # hide
```

Now let's define a neural network with a `NeuralODE` layer. First we define
the layer. Here we're going to use `FastChain`, which is a faster neural network
structure for NeuralODEs:

```@example nnode
dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
nothing # hide
```

Note that we can directly use `Chain`s from Flux.jl as well, for example:

```
dudt2 = Chain(x -> x.^3,
              Dense(2, 50, tanh),
              Dense(50, 2))
```

In our model we used the `x -> x.^3` assumption in the model. By incorporating
structure into our equations, we can reduce the required size and training time
for the neural network, but a good guess needs to be known!

From here we build a loss function around it. The `NeuralODE` has an optional
second argument for new parameters which we will use to iteratively change the
neural network in our training loop. We will use the L2 loss of the network's
output against the time series data:

```@example nnode
function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
nothing # hide
```

We define a callback function.

```@example nnode
# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred; doplot = false)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  display(l)

  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1,:], label = "data")
  scatter!(plt, tsteps, pred[1,:], label = "prediction")
  push!(list_plots, plt)
  if doplot
    display(plot(plt))
  end

  return false
end
nothing # hide
```

We then train the neural network to learn the ODE.

Here we showcase starting the optimization with `ADAM` to more quickly find a
minimum, and then honing in on the minimum by using `LBFGS`. By using the two
together, we are able to fit the neural ODE in 9 seconds! (Note, the timing
commented out the plotting).

```@example nnode
# Train using the ADAM optimizer
result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 300)
```

```@example nnode
animate(list_plots, "NN_anim.gif"); nothing # hide
```

![Neural Network ODE animation](NN_anim.gif)

We then complete the training using a different optimizer starting from where
`ADAM` stopped.

```@example nnode
# Retrain using the LBFGS optimizer
result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode.minimizer,
                                           LBFGS(),
                                           cb = callback)
```

```@example nnode
animate(list_plots, "NN_anim2.gif"); nothing # hide
```

![Neural Network ODE animation](NN_anim2.gif)
