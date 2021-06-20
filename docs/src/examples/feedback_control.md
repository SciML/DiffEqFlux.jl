# Universal Differential Equations for Neural Feedback Control

You can also mix a known differential equation and a neural differential
equation, so that the parameters and the neural network are estimated
simultaneously!

We will assume that we know the dynamics of the second equation
(linear dynamics), and our goal is to find a neural network that is dependent
on the current state of the dynamical system that will control the second
equation to stay close to 1.

```julia
using DiffEqFlux, DifferentialEquations, Plots

u0 = 1.1f0
tspan = (0.0f0, 25.0f0)
tsteps = 0.0f0:1.0:25.0f0

model_univ = FastChain(FastDense(2, 16, tanh),
                       FastDense(16, 16, tanh),
                       FastDense(16, 1))

# The model weights are destructured into a vector of parameters
p_model = initial_params(model_univ)
n_weights = length(p_model)

# Parameters of the second equation (linear dynamics)
p_system = Float32[0.5, -0.5]

p_all = [p_model; p_system]
θ = Float32[u0; p_all]

function dudt_univ!(du, u, p, t)
    # Destructure the parameters
    model_weights = p[1:n_weights]
    α = p[end - 1]
    β = p[end]

    # The neural network outputs a control taken by the system
    # The system then produces an output
    model_control, system_output = u

    # Dynamics of the control and system
    dmodel_control = model_univ(u, model_weights)[1]
    dsystem_output = α*system_output + β*model_control

    # Update in place
    du[1] = dmodel_control
    du[2] = dsystem_output
end

prob_univ = ODEProblem(dudt_univ!, [0f0, u0], tspan, p_all)
sol_univ = solve(prob_univ, Tsit5(),abstol = 1e-8, reltol = 1e-6)

function predict_univ(θ)
  return Array(solve(prob_univ, Tsit5(), u0=[0f0, θ[1]], p=θ[2:end],
                              saveat = tsteps))
end

loss_univ(θ) = sum(abs2, predict_univ(θ)[2,:] .- 1)
l = loss_univ(θ)
```

```julia
list_plots = []
iter = 0
callback = function (θ, l)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  println(l)

  plt = plot(predict_univ(θ)', ylim = (0, 6))
  push!(list_plots, plt)
  display(plt)
  return false
end
```

```julia
result_univ = DiffEqFlux.sciml_train(loss_univ, θ,
                                     BFGS(initial_stepnorm = 0.01),
                                     cb = callback,
                                     allow_f_increases = false)
```

```
* Status: success

* Candidate solution
   u: [1.00e+00, 4.33e-02, 3.72e-01,  ...]
   Minimum:   6.572520e-13

* Found with
   Algorithm:     BFGS
   Initial Point: [1.10e+00, 4.18e-02, 3.64e-01,  ...]

* Convergence measures
   |x - x'|               = 0.00e+00 ≤ 0.0e+00
   |x - x'|/|x'|          = 0.00e+00 ≤ 0.0e+00
   |f(x) - f(x')|         = 0.00e+00 ≤ 0.0e+00
   |f(x) - f(x')|/|f(x')| = 0.00e+00 ≤ 0.0e+00
   |g(x)|                 = 5.45e-06 ≰ 1.0e-08

* Work counters
   Seconds run:   8  (vs limit Inf)
   Iterations:    23
   f(x) calls:    172
   ∇f(x) calls:   172
```

Notice that in just 23 iterations or 8 seconds we get to a minimum of `7e-13`,
successfully solving the nonlinear optimal control problem.
