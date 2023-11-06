# Neural Ordinary Differential Equations

A neural ODE is an ODE where a neural
network defines its derivative function. For example, with the multilayer
perceptron neural network `Lux.Chain(Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))`,
we can define a differential equation which is `u' = NN(u)`. This is done simply
by the `NeuralODE` struct. Let's take a look at an example.

## Copy-Pasteable Code

Before getting to the explanation, here's some code to start with. We will
follow a full explanation of the definition and training process:

```@example neuralode_cp
using ComponentArrays, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots

rng = Random.default_rng()
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

dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = false)
  println(l)
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                       Adam(0.05),
                                       callback = callback,
                                       maxiters = 300)

optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
```

![Neural ODE](https://user-images.githubusercontent.com/1814174/88589293-e8207f80-d026-11ea-86e2-8a3feb8252ca.gif)

## Explanation

Let's get a time series array from a spiral ODE to train against.

```@example neuralode
using ComponentArrays, Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots

rng = Random.default_rng()
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
```

Now let's define a neural network with a `NeuralODE` layer. First, we define
the layer. Here we're going to use `Lux.Chain`, which is a suitable neural network
structure for NeuralODEs with separate handling of state variables:

```@example neuralode
dudt2 = Lux.Chain(x -> x.^3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
```

Note that we can directly use `Chain`s from Flux.jl as well, for example:

```julia
dudt2 = Chain(x -> x.^3,
              Dense(2, 50, tanh),
              Dense(50, 2))
```

In our model, we used the `x -> x.^3` assumption in the model. By incorporating
structure into our equations, we can reduce the required size and training time
for the neural network, but a good guess needs to be known!

From here we build a loss function around it. The `NeuralODE` has an optional
second argument for new parameters, which we will use to change the
neural network iteratively in our training loop. We will use the L2 loss of the network's
output against the time series data:

```@example neuralode
function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
```

We define a callback function. In this example, we set `doplot = false` because otherwise
it would show every step and overflow the documentation, but for your use case
**set doplot=true to see a live animation of the training process!**

```@example neuralode
# Callback function to observe training
callback = function (p, l, pred; doplot = false)
  println(l)
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...)
```

We then train the neural network to learn the ODE.

Here we showcase starting the optimization with `Adam` to more quickly find a
minimum, and then honing in on the minimum by using `LBFGS`. By using the two
together, we can fit the neural ODE in 9 seconds! (Note, the timing
commented out the plotting). You can easily incorporate the procedure below to
set up custom optimization problems. For more information on the usage of
[Optimization.jl](https://github.com/SciML/Optimization.jl), please consult
[this](https://docs.sciml.ai/Optimization/stable/) documentation.

The `x` and `p` variables in the optimization function are different from
`x` and `p` above. The optimization function runs over the space of parameters of
the original problem, so `x_optimization` == `p_original`.

```@example neuralode
# Train using the Adam optimizer
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                       Adam(0.05),
                                       callback = callback,
                                       maxiters = 300)
```

We then complete the training using a different optimizer, starting from where
`Adam` stopped. We do `allow_f_increases=false` to make the optimization automatically
halt when near the minimum.

```@example neuralode
# Retrain using the LBFGS optimizer
optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback = callback,
                                        allow_f_increases = false)
```

And then we use the callback with `doplot=true` to see the final plot:

```@example neuralode
callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)
```
