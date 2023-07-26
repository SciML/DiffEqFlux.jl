# Continuous Normalizing Flows

Now, we study a single layer neural network that can estimate the density `p_x` of a variable of interest `x` by re-parameterizing a base variable `z` with known density `p_z` through the Neural Network model passed to the layer.

## Copy-Pasteable Code

Before getting to the explanation, here's some code to start with. We will
follow a full explanation of the definition and training process:

```@example cnf
using Flux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationFlux,
      OptimizationOptimJL, Distributions

nn = Flux.Chain(
    Flux.Dense(1, 3, tanh),
    Flux.Dense(3, 1, tanh),
) |> f32
tspan = (0.0f0, 10.0f0)

ffjord_mdl = FFJORD(nn, tspan, Tsit5())

# Training
data_dist = Normal(6.0f0, 0.7f0)
train_data = Float32.(rand(data_dist, 1, 100))

function loss(θ)
    logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ)
    -mean(logpx)
end

function cb(p, l)
    @info "Training" loss = loss(p)
    false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ffjord_mdl.p)

res1 = Optimization.solve(optprob,
                          Adam(0.1),
                          maxiters = 100,
                          callback=cb)

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2,
                          Optim.LBFGS(),
                          allow_f_increases=false,
                          callback=cb)

# Evaluation
using Distances

actual_pdf = pdf.(data_dist, train_data)
learned_pdf = exp.(ffjord_mdl(train_data, res2.u, monte_carlo=false)[1])
train_dis = totalvariation(learned_pdf, actual_pdf) / size(train_data, 2)

# Data Generation
ffjord_dist = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p=res2.u))
new_data = rand(ffjord_dist, 100)
```

## Step-by-Step Explanation

We can use DiffEqFlux.jl to define, train and output the densities computed by CNF layers. In the same way as a neural ODE, the layer takes a neural network that defines its derivative function (see [1] for a reference). A possible way to define a CNF layer, would be:

```@example cnf2
using Flux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationFlux,
      OptimizationOptimJL, Distributions

nn = Flux.Chain(
    Flux.Dense(1, 3, tanh),
    Flux.Dense(3, 1, tanh),
) |> f32
tspan = (0.0f0, 10.0f0)

ffjord_mdl = FFJORD(nn, tspan, Tsit5())
```

where we also pass as an input the desired timespan for which the differential equation that defines `log p_x` and `z(t)` will be solved.

### Training

First, let's get an array from a normal distribution as the training data. Note that we want the data in Float32
values to match how we have set up the neural network weights and the state space of the ODE.

```@example cnf2
data_dist = Normal(6.0f0, 0.7f0)
train_data = Float32.(rand(data_dist, 1, 100))
```

Now we define a loss function that we wish to minimize and a callback function to track loss improvements

```@example cnf2
function loss(θ)
    logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ)
    -mean(logpx)
end

function cb(p, l)
    @info "Training" loss = loss(p)
    false
end
```

In this example, we wish to choose the parameters of the network such that the likelihood of the re-parameterized variable is maximized. Other loss functions may be used depending on the application. Furthermore, the CNF layer gives the log of the density of the variable x, as one may guess from the code above.

We then train the neural network to learn the distribution of `x`.

Here we showcase starting the optimization with `Adam` to more quickly find a minimum, and then honing in on the minimum by using `LBFGS`.

```@example cnf2
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ffjord_mdl.p)

res1 = Optimization.solve(optprob,
                          Adam(0.1),
                          maxiters = 100,
                          callback=cb)
```

We then complete the training using a different optimizer, starting from where `Adam` stopped.

```@example cnf2
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2,
                          Optim.LBFGS(),
                          allow_f_increases=false,
                          callback=cb)
```

### Evaluation

For evaluating the result, we can use `totalvariation` function from `Distances.jl`. First, we compute densities using actual distribution and FFJORD model.
Then we use a distance function between these distributions.

```@example cnf2
using Distances

actual_pdf = pdf.(data_dist, train_data)
learned_pdf = exp.(ffjord_mdl(train_data, res2.u, monte_carlo=false)[1])
train_dis = totalvariation(learned_pdf, actual_pdf) / size(train_data, 2)
```

### Data Generation

What's more, we can generate new data by using FFJORD as a distribution in `rand`.

```@example cnf2
ffjord_dist = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p=res2.u))
new_data = rand(ffjord_dist, 100)
```

## References

[1] Grathwohl, Will, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. "Ffjord: Free-form continuous dynamics for scalable reversible generative models." arXiv preprint arXiv:1810.01367 (2018).
