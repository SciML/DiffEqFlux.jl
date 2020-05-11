# Delay Differential Equations

Other differential equation problem types from DifferentialEquations.jl are
supported. For example, we can build a layer with a delay differential equation
like:

```
using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity

# Define the same LV equation, but including a delay parameter
function delay_lotka_volterra!(du, u, h, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α   - β*y) * h(p, t-0.1)[1]
  du[2] = dy = (δ*x - γ)   * y
end

# Initial parameters
p = [2.2, 1.0, 2.0, 0.4]

# Define a vector containing delays for each variable (although only the first
# one is used)
h(p, t) = ones(eltype(p), 2)

# Initial conditions
u0 = [1.0, 1.0]

# Define the problem as a delay differential equation
prob_dde = DDEProblem(delay_lotka_volterra!, u0, h, (0.0, 10.0),
                      constant_lags = [0.1])

function predict_dde(p)
  return Array(concrete_solve(prob_dde, MethodOfSteps(Tsit5()),
                              u0, p, saveat = 0.1,
                              sensealg = TrackerAdjoint()))
end

loss_dde(p) = sum(abs2, x-1 for x in predict_dde(p))
```

Notice that we chose `sensealg = TrackerAdjoint()` to utilize the Tracker.jl
reverse-mode to handle the delay differential equation.
