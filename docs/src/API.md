# API Documentation

## Neural DE Layer Functions

- `NeuralODE(model, tspan, solver, args...; kwargs...)` defines a neural ODE
  layer where `model` is a Flux.jl model, `tspan` is the time span to integrate,
  and the rest of the arguments are passed to the ODE solver.

- `NeuralODEMM(model, constraints, tspan, mass_matrix, args...; kwargs...)`
  defines a neural ODE layer with a mass matrix, i.e. `Mu=[model(u);
  constraints(u, p, t)]` where the constraints cover the rank-deficient area of
  the mass matrix (i.e., the constraints of a differential-algebraic equation).
  Thus the mass matrix is allowed to be singular.

- `NeuralDSDE(model1, model2, tspan, solver, args...; kwargs...)` defines a
  neural SDE layer where `model1` is a Flux.jl for the drift equation, `model2`
  is a Flux.jl model for the diffusion equation, `tspan` is the time span to
  integrate, and the rest of the arguments are passed to the SDE solver. The
  noise is diagonal, i.e. it assumes a vector output and performs `model2(u) .*
  dW` against a dW matching the number of states.

- `NeuralSDE(model1, model2, tspan, nbrown, solver, args...; kwargs...)` defines
  a neural SDE layer where `model1` is a Flux.jl for the drift equation,
  `model2` is a Flux.jl model for the diffusion equation, `tspan` is the time
  span to integrate, `nbrown` is the number of Brownian motions, and the rest of
  the arguments are passed to the SDE solver. The model is multiplicative, i.e.
  it's interpreted as `model2(u) * dW`, and so the return of `model2` should be
  an appropriate matrix for performing this multiplication, i.e. the size of its
  output should be `length(x) x nbrown`.

- `NeuralCDDE(model, tspan, lags, solver, args...; kwargs...)`defines a neural
  DDE layer where `model` is a Flux.jl model, `tspan` is the time span to
  integrate, lags is the lagged values to use in the predictor, and the rest of
  the arguments are passed to the ODE solver. The model should take in a vector
  that concatenates the lagged states, i.e. `[u(t); u(t-lags[1]); ...; u(t -
  lags[end])]`
