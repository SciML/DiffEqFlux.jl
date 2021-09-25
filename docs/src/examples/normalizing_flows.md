# Continuous Normalizing Flows with GalacticOptim.jl

Now, we study a single layer neural network that can estimate the density `p_x` of a variable of interest `x` by re-parameterizing a base variable `z` with known density `p_z` through the Neural Network model passed to the layer.

## Copy-Pasteable Code

Before getting to the explanation, here's some code to start with. We will
follow a full explanation of the definition and training process:

```julia
using DiffEqFlux, DifferentialEquations, GalacticOptim, Distributions

nn = Chain(
    Dense(1, 3, tanh),
    Dense(3, 1, tanh),
) |> f32
tspan = (0.0f0, 10.0f0)
ffjord_mdl = FFJORD(nn, tspan, Tsit5())

# Training
data_dist = Normal(6.0f0, 0.7f0)
train_data = rand(data_dist, 1, 100)

function loss(θ)
    logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ)
    -mean(logpx)
end

adtype = GalacticOptim.AutoZygote()
res1 = DiffEqFlux.sciml_train(loss, ffjord_mdl.p, ADAM(0.1), adtype; maxiters=100)
res2 = DiffEqFlux.sciml_train(loss, res1.u, LBFGS(), adtype; allow_f_increases=false)

# Evaluation
using Distances

actual_pdf = pdf.(data_dist, train_data)
learned_pdf = exp.(ffjord_mdl(train_data, res2.u)[1])
train_dis = totalvariation(learned_pdf, actual_pdf) / size(train_data, 2)

# Data Generation
ffjord_dist = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p=res2.u))
new_data = rand(ffjord_dist, 100)
```

We can use DiffEqFlux.jl to define, train and output the densities computed by CNF layers. In the same way as a neural ODE, the layer takes a neural network that defines its derivative function (see [1] for a reference). A possible way to define a CNF layer, would be:

```julia
using DiffEqFlux, DifferentialEquations, GalacticOptim, Distributions

nn = Chain(
    Dense(1, 3, tanh),
    Dense(3, 1, tanh),
) |> f32
tspan = (0.0f0, 10.0f0)
ffjord_mdl = FFJORD(nn, tspan, Tsit5())
```

where we also pass as an input the desired timespan for which the differential equation that defines `log p_x` and `z(t)` will be solved.

### Training

First, let's get an array from a normal distribution as the training data

```julia
data_dist = Normal(6.0f0, 0.7f0)
train_data = rand(data_dist, 1, 100)
```

Now we define a loss function that we wish to minimize

```julia
function loss(θ)
    logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ)
    -mean(logpx)
end
```

In this example, we wish to choose the parameters of the network such that the likelihood of the re-parameterized variable is maximized. Other loss functions may be used depending on the application. Furthermore, the CNF layer gives the log of the density of the variable x, as one may guess from the code above.

We then train the neural network to learn the distribution of `x`.

Here we showcase starting the optimization with `ADAM` to more quickly find a minimum, and then honing in on the minimum by using `LBFGS`.

```julia
adtype = GalacticOptim.AutoZygote()
res1 = DiffEqFlux.sciml_train(loss, ffjord_mdl.p, ADAM(0.1), adtype; maxiters=100)

# output
* Status: success

* Candidate solution
   u: [-1.88e+00, 2.44e+00, 2.01e-01,  ...]
   Minimum:   1.240627e+00

* Found with
   Algorithm:     ADAM
   Initial Point: [9.33e-01, 1.13e+00, 2.92e-01,  ...]

```

We then complete the training using a different optimizer starting from where `ADAM` stopped.

```julia
res2 = DiffEqFlux.sciml_train(loss, res1.u, LBFGS(), adtype; allow_f_increases=false)

# output
* Status: success

* Candidate solution
   u: [-1.06e+00, 2.24e+00, 8.77e-01,  ...]
   Minimum:   1.157672e+00

* Found with
   Algorithm:     L-BFGS
   Initial Point: [-1.88e+00, 2.44e+00, 2.01e-01,  ...]

* Convergence measures
   |x - x'|               = 0.00e+00 ≰ 0.0e+00
   |x - x'|/|x'|          = 0.00e+00 ≰ 0.0e+00
   |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
   |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
   |g(x)|                 = 4.09e-03 ≰ 1.0e-08

* Work counters
   Seconds run:   514  (vs limit Inf)
   Iterations:    44
   f(x) calls:    244
   ∇f(x) calls:   244
```

### Evaluation

For evaluating the result, we can use `totalvariation` function from `Distances.jl`. First, we compute densities using actual distribution and FFJORD model. then we use a distance function.

```julia
using Distances

actual_pdf = pdf.(data_dist, train_data)
learned_pdf = exp.(ffjord_mdl(train_data, res2.u)[1])
train_dis = totalvariation(learned_pdf, actual_pdf) / size(train_data, 2)
```

### Data Generation

What's more, we can generate new data by using FFJORD as a distribution in `rand`.

```julia
ffjord_dist = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p=res2.u))
new_data = rand(ffjord_dist, 100)
```

`References`:

[1] W. Grathwohl, R. T. Q. Chen, J. Bettencourt, I. Sutskever, D. Duvenaud. FFJORD: Free-Form Continuous Dynamic For Scalable Reversible Generative Models. arXiv preprint at arXiv1810.01367, 2018.
