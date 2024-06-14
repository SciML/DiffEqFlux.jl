# Enforcing Physical Constraints via Universal Differential-Algebraic Equations

As shown in [DiffEqDocs](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/dae_example/),
differential-algebraic equations (DAEs) can be used to impose physical
constraints. One way to define a DAE is through an ODE with a singular mass
matrix. For example, if we make `Mu' = f(u)` where the last row of `M` is all
zeros, then we have a constraint defined by the right-hand side. Using
`NeuralODEMM`, we can use this to define a neural ODE where the sum of all 3
terms must add to one. An example of this is as follows:

```@example dae
using DiffEqFlux
using Lux, ComponentArrays, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Plots

using Random
rng = Random.default_rng()

function f!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    return nothing
end

u₀ = [1.0, 0, 0]
M = [1.0 0 0
     0 1.0 0
     0 0 0]

tspan = (0.0, 1.0)
p = [0.04, 3e7, 1e4]

stiff_func = ODEFunction(f!; mass_matrix = M)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(); saveat = 0.1)

nn_dudt2 = Lux.Chain(Lux.Dense(3, 64, tanh), Lux.Dense(64, 2))

pinit, st = Lux.setup(rng, nn_dudt2)

model_stiff_ndae = NeuralODEMM(nn_dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1],
    tspan, M, Rodas5(; autodiff = false); saveat = 0.1)

function predict_stiff_ndae(p)
    return model_stiff_ndae(u₀, p, st)[1]
end

function loss_stiff_ndae(p)
    pred = predict_stiff_ndae(p)
    loss = sum(abs2, Array(sol_stiff) .- pred)
    return loss, pred
end

# callback = function (state, l, pred) #callback function to observe training
#   display(l)
#   return false
# end

l1 = first(loss_stiff_ndae(ComponentArray(pinit)))

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_stiff_ndae(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(pinit))
result_stiff = Optimization.solve(optprob, OptimizationOptimJL.BFGS(); maxiters = 100)
```

## Step-by-Step Description

### Load Packages

```@example dae2
using DiffEqFlux
using Lux, ComponentArrays, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Plots

using Random
rng = Random.default_rng()
```

### Differential Equation

First, we define our differential equations as a highly stiff problem, which makes the
fitting difficult.

```@example dae2
function f!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    return nothing
end
```

### Parameters

```@example dae2
u₀ = [1.0, 0, 0]

M = [1.0 0 0
     0 1.0 0
     0 0 0]

tspan = (0.0, 1.0)

p = [0.04, 3e7, 1e4]
```

  - `u₀` = Initial Conditions
  - `M` = Semi-explicit Mass Matrix (last row is the constraint equation and are therefore
    all zeros)
  - `tspan` = Time span over which to evaluate
  - `p` = parameters `k1`, `k2` and `k3` of the differential equation above

### ODE Function, Problem and Solution

We define and solve our ODE problem to generate the “labeled” data which will be used to
train our Neural Network.

```@example dae2
stiff_func = ODEFunction(f!; mass_matrix = M)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(); saveat = 0.1)
```

Because this is a DAE, we need to make sure to use a **compatible solver**.
`Rodas5` works well for this example.

### Neural Network Layers

Next, we create our layers using `Lux.Chain`. We use this instead of `Flux.Chain` because it
is more suited to SciML applications (similarly for `Lux.Dense`). The input to our network
will be the initial conditions fed in as `u₀`.

```@example dae2
nn_dudt2 = Lux.Chain(Lux.Dense(3, 64, tanh), Lux.Dense(64, 2))

pinit, st = Lux.setup(rng, nn_dudt2)

model_stiff_ndae = NeuralODEMM(nn_dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1],
    tspan, M, Rodas5(; autodiff = false); saveat = 0.1)
model_stiff_ndae(u₀, ComponentArray(pinit), st)
```

Because this is a stiff problem, we have manually imposed that sum constraint via
`(u,p,t) -> [u[1] + u[2] + u[3] - 1]`, making the fitting easier.

### Prediction Function

For simplicity, we define a wrapper function that only takes in the model's parameters
to make predictions.

```@example dae2
function predict_stiff_ndae(p)
    return model_stiff_ndae(u₀, p, st)[1]
end
```

### Train Parameters

Training our network requires a **loss function**, an **optimizer**, and a
**callback function** to display the progress.

#### Loss

We first make our predictions based on the current parameters, then calculate the loss
from these predictions. In this case, we use **least squares** as our loss.

```@example dae2
function loss_stiff_ndae(p)
    pred = predict_stiff_ndae(p)
    loss = sum(abs2, sol_stiff .- pred)
    return loss, pred
end

l1 = first(loss_stiff_ndae(ComponentArray(pinit)))
```

Notice that we are feeding the **parameters** of `model_stiff_ndae` to the `loss_stiff_ndae`
function. `model_stiff_node.p` are the weights of our NN and is of size *386*
(4 * 64 + 65 * 2) including the biases.

#### Optimizer

The optimizer is `BFGS`(see below).

#### Callback

The callback function displays the loss during training.

```@example dae2
callback = function (state, l, pred) #callback function to observe training
    display(l)
    return false
end
```

### Train

Finally, training with `Optimization.solve` by passing: *loss function*, *model parameters*,
*optimizer*, *callback* and *maximum iteration*.

```@example dae2
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_stiff_ndae(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(pinit))
result_stiff = Optimization.solve(optprob, OptimizationOptimJL.BFGS(); maxiters = 100)
```
