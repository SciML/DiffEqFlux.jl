# Physics Informed Machine Learning with TensorLayer

In this tutorial, we show how to use the `DiffEqFlux` TensorLayer to solve problems
in Physics Informed Machine Learning.

Let's consider the anharmonic oscillator described by the ODE

```math
ẍ = - kx - αx³ - βẋ -γẋ³.
```

To obtain the training data, we solve the equation of motion using one of the
solvers in `DifferentialEquations`:

```julia
using DiffEqFlux, DifferentialEquations, LinearAlgebra
k, α, β, γ = 1, 0.1, 0.2, 0.3
tspan = (0.0,10.0)

function dxdt_train(du,u,p,t)
  du[1] = u[2]
  du[2] = -k*u[1] - α*u[1]^3 - β*u[2] - γ*u[2]^3
end

u0 = [1.0,0.0]
ts = collect(0.0:0.1:tspan[2])
prob_train = ODEProblem{true}(dxdt_train,u0,tspan,p=nothing)
data_train = Array(solve(prob_train,Tsit5(),saveat=ts))
```

Now, we create a TensorLayer that will be able to perform 10th order expansions in
a Legendre Basis:

```julia
A = [LegendreBasis(10), LegendreBasis(10)]
nn = TensorLayer(A, 1)
```

and we also instantiate the model we are trying to learn, "informing" the neural
about the `∝x` and `∝v` dependencies in the equation of motion:

```julia
f = x -> min(30one(x),x)

function dxdt_pred(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]*u[1] - p[2]*u[2] + f(nn(u,p[3:end])[1])
end

α = zeros(102)

prob_pred = ODEProblem{true}(dxdt_pred,u0,tspan,p=nothing)
```

Note that we introduced a "cap" in the neural network term to avoid instabilities
in the solution of the ODE. We also initialized the vector of parameters to zero
in order to obtain a faster convergence for this particular example.

Finally, we introduce the corresponding loss function:

```julia

function predict_adjoint(θ)
  x = Array(solve(prob_pred,Tsit5(),p=θ,saveat=ts))
end

function loss_adjoint(θ)
  x = predict_adjoint(θ)
  loss = sum(norm.(x - data_train))
  return loss
end

function cb(θ,l)
  @show θ, l
  return false
end
```

and we train the network using two rounds of `ADAM`:

```julia
res1 = DiffEqFlux.sciml_train(loss_adjoint, α, ADAM(0.05), cb = cb, maxiters = 150)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.minimizer, ADAM(0.001), cb = cb,maxiters = 150)
opt = res2.minimizer
```

We plot the results and we obtain a fairly accurate learned model:

```julia
using Plots
data_pred = predict_adjoint(opt)
plot(ts, data_train[1,:], label = "X (ODE)")
plot!(ts, data_train[2,:], label = "V (ODE)")
plot!(ts, data_pred[1,:], label = "X (NN)")
plot!(ts, data_pred[2,:],label = "V (NN)")
```

![plot_tutorial](https://user-images.githubusercontent.com/61364108/85925795-e2d5e680-b868-11ea-9816-29f8125c8cb5.png)
