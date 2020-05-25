# Enforcing Physical Constraints via Universal Differential-Algebraic Equations

As shown in the [stiff ODE tutorial](https://docs.juliadiffeq.org/latest/tutorials/advanced_ode_example/#Handling-Mass-Matrices-1),
differential-algebraic equations (DAEs) can be used to impose physical
constraints. One way to define a DAE is through an ODE with a singular mass
matrix. For example, if we make `Mu' = f(u)` where the last row of `M` is all
zeros, then we have a constraint defined by the right hand side. Using
`NeuralODEMM`, we can use this to define a neural ODE where the sum of all 3
terms must add to one. An example of this is as follows:

```julia
using Flux, DiffEqFlux, OrdinaryDiffEq, Optim, Test, Plots

# A desired MWE for now, not a test yet.
function f!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2
    du[3] =  y₁ + y₂ + y₃ - 1
    return nothing
end

u₀ = [1.0, 0, 0]
M = [1. 0  0
     0  1. 0
     0  0  0]

tspan = (0.0,1.0)
p = [0.04, 3e7, 1e4]

stiff_func = ODEFunction(f!, mass_matrix = M)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(), saveat = 0.1)

nn_dudt2 = FastChain(FastDense(3, 64, tanh),
                     FastDense(64, 2))

model_stiff_ndae = NeuralODEMM(nn_dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1],
                               tspan, M, Rodas5(autodiff = false), saveat = 0.1)
model_stiff_ndae(u₀)

function predict_stiff_ndae(p)
    return model_stiff_ndae(u₀, p)
end

function loss_stiff_ndae(p)
    pred = predict_stiff_ndae(p)
    loss = sum(abs2, sol_stiff .- pred)
    return loss, pred
end

callback = function (p, l, pred) #callback function to observe training
  display(l)
  return false
end

l1 = first(loss_stiff_ndae(model_stiff_ndae.p))
result_stiff = DiffEqFlux.sciml_train(loss_stiff_ndae, model_stiff_ndae.p,
                                      BFGS(initial_stepnorm = 0.001),
                                      cb = callback, maxiters = 100)
```


## Step-by-Step Description

### Load Packages

```julia
using Flux, DiffEqFlux, OrdinaryDiffEq, Optim, Test, Plots
```

### Differential Equation
```Julia
# A desired MWE for now, not a test yet.
function f!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2
    du[3] =  y₁ + y₂ + y₃ - 1
    return nothing
end
```

### Parameters
```Julia
u₀ = [1.0, 0, 0]

M = [1. 0  0
     0  1. 0
     0  0  0]

tspan = (0.0,1.0)

p = [0.04, 3e7, 1e4]
```

### ODE Function, Problem and Solution
```Julia
stiff_func = ODEFunction(f!, mass_matrix = M)
prob_stiff = ODEProblem(stiff_func, u₀, tspan, p)
sol_stiff = solve(prob_stiff, Rodas5(), saveat = 0.1)
```

### Neural Network Layers

This is a highly stiff problem, making the fitting difficult, but we have
manually imposed that sum constraint via `(u,p,t) -> [u[1] + u[2] + u[3] - 1]`,
making the fitting easier.

```Julia
nn_dudt2 = FastChain(FastDense(3, 64, tanh),
                     FastDense(64, 2))

model_stiff_ndae = NeuralODEMM(nn_dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1],
                               tspan, M, Rodas5(autodiff = false), saveat = 0.1)
model_stiff_ndae(u₀)
```

### Prediction
```Julia
function predict_stiff_ndae(p)
    return model_stiff_ndae(u₀, p)
end
```

### Train Parameters

#### Loss
```Julia
function loss_stiff_ndae(p)
    pred = predict_stiff_ndae(p)
    loss = sum(abs2, sol_stiff .- pred)
    return loss, pred
end

l1 = first(loss_stiff_ndae(model_stiff_ndae.p))
```

#### Callback
```Julia
callback = function (p, l, pred) #callback function to observe training
  display(l)
  return false
end
```

### Train

```Julia
result_stiff = DiffEqFlux.sciml_train(loss_stiff_ndae, model_stiff_ndae.p,
                                      BFGS(initial_stepnorm = 0.001),
                                      cb = callback, maxiters = 100)
```

