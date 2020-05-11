# Stochastic Differential Equation

Here we demonstrate `sensealg = ForwardDiffSensitivity()` (provided by
DiffEqSensitivity.jl) for forward-mode automatic differentiation of a small
stochastic differential equation. For large parameter equations, like neural
stochastic differential equations, you should use reverse-mode automatic
differentition. However, forward-mode can be more efficient for low numbers
of parameters (<100).

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

function lotka_volterra_noise!(du, u, p, t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end

u0 = [1.0,1.0]
tspan = (0.0, 10.0)
p = [2.2, 1.0, 2.0, 0.4]
prob_sde = SDEProblem(lotka_volterra!, lotka_volterra_noise!, u0, tspan)


function predict_sde(p)
  return Array(concrete_solve(prob_sde, SOSRI(), u0, p,
               sensealg = ForwardDiffSensitivity(), saveat = 0.1))
end

loss_sde(p) = sum(abs2, x-1 for x in predict_sde(p))
nothing
```

For this training process, because the loss function is stochastic, we will use
the `ADAM` optimizer from Flux.jl. The `sciml_train` function is the same as
before. However, to speed up the training process, we will use a global counter
so that way we only plot the current results every 10 iterations. This looks
like:

```julia
list_plots = []
iter = 0
callback = function (p, l)
  global list_plots, iter

  # List plots is reset to an empty list on the first callback
  if iter == 0
    list_plots = []
  end
  iter += 1

  display(l)

  if iter%10 == 1
    remade_solution = solve(remake(prob_sde, p = p), SOSRI(), saveat = 0.1)
    plt = plot(remade_solution, ylim = (0, 6))
    push!(list_plots, plt)
    display(plt)
  end
  return false
end
nothing # hide
```

Let's optimise

```julia
result_sde = DiffEqFlux.sciml_train(loss_sde, p, ADAM(0.1),
                                    cb = callback, maxiters = 100)
```

![](https://user-images.githubusercontent.com/1814174/51399524-2c6abf80-1b14-11e9-96ae-0192f7debd03.gif)
