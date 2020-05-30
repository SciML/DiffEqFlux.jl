# Optimization of Ordinary Differential Equations

First let's create a Lotka-Volterra ODE using DifferentialEquations.jl. For
more details, [see the DifferentialEquations.jl documentation](http://docs.juliadiffeq.org/dev/). The Lotka-Volterra equations have the form:

```math
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta x y      \\
\frac{dy}{dt} &= -\delta y + \gamma x y    \\
\end{aligned}
```

```julia
using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqSensitivity, Plots

function lotka_volterra!(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

# Initial condition
u0 = [1.0, 1.0]

# Simulation interval and intermediary points
tspan = (0.0, 10.0)
tsteps = 0.0:0.1:10.0

# LV equation parameter. p = [α, β, δ, γ]
p = [1.5, 1.0, 3.0, 1.0]

# Setup the ODE problem, then solve
prob_ode = ODEProblem(lotka_volterra!, u0, tspan, p)
sol_ode = solve(prob_ode, Tsit5())

# Plot the solution
using Plots
plot(sol_ode)
savefig("LV_ode.png")
```

![LV Solution Plot](https://user-images.githubusercontent.com/1814174/51388169-9a07f300-1af6-11e9-8c6c-83c41e81d11c.png)

For this first example, we do not yet include a neural network. We take
[AD-compatible `solve`
function](https://docs.juliadiffeq.org/latest/analysis/sensitivity/) function
that takes the parameters and an initial condition and returns the solution of
the differential equation as a
[`DiffEqArray`](https://github.com/JuliaDiffEq/RecursiveArrayTools.jl) (same
array semantics as the standard differential equation solution object but
without the interpolations).

```julia
# Create a solution (prediction) for a given starting point u0 and set of
# parameters p
function predict_adjoint(p)
  return Array(solve(prob_ode, Tsit5(), p=p, saveat = tsteps))
end
nothing
```

Next we choose a square loss function. Our goal will be to find parameter that
make the Lotka-Volterra solution constant `x(t)=1`, so we defined our loss as
the squared distance from 1. Note that when using `sciml_train`, the first
return is the loss value, and the other returns are sent to the callback for
monitoring convergence.

```julia
function loss_adjoint(p)
  prediction = predict_adjoint(p)
  loss = sum(abs2, x-1 for x in prediction)
  return loss, prediction
end
nothing
```

Lastly, we use the `sciml_train` function to train the parameters using BFGS to
arrive at parameters which optimize for our goal. `sciml_train` allows defining
a callback that will be called at each step of our training loop. It takes in
the current parameter vector and the returns of the last call to the loss
function. We will display the current loss and make a plot of the current
situation:

```julia
# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred)
  global iter, list_plots

  if iter == 0
    list_plots = []
  end
  iter += 1

  display(l)

  # using `remake` to re-create our `prob` with current parameters `p`
  remade_solution = solve(remake(prob_ode, p = p), Tsit5(), saveat = tsteps)
  plt = plot(remade_solution, ylim = (0, 6))

  push!(list_plots, plt)
  display(plt)

  # Tell sciml_train to not halt the optimization. If return true, then
  # optimization stops.
  return false
end
nothing
```

Let's optimise the model.

```julia
result_ode = DiffEqFlux.sciml_train(loss_adjoint, p,
                                    BFGS(initial_stepnorm = 0.0001),
                                    cb = callback)
```

In just seconds we found parameters which give a relative loss of `1e-6`! We can
get the final loss with `result_ode.minimum`, and get the optimal parameters
with `result_ode.minimizer`. For example, we can plot the final outcome and show
that we solved the control problem and successfully found parameters to make the
ODE solution constant:

```julia
remade_solution = solve(remake(prob_ode, p = result_ode.minimizer), Tsit5(),      
                        saveat = tsteps)
#plot(remade_solution, ylim = (0, 6))
#savefig("LV_ode2.png")
```

![Final plot](https://user-images.githubusercontent.com/1814174/51399500-1f4dd080-1b14-11e9-8c9d-144f93b6eac2.gif)

This shows the evolution of the solutions:

```@example ode
animate(list_plots) # hide
```
