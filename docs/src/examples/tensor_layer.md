# Physics Informed Machine Learning with TensorLayer

In this tutorial, we show how to use the DiffEqFlux TensorLayer to solve problems
in Physics Informed Machine Learning.

Let's consider the anharmonic oscillator described by the ODE

ẍ = - kx - αx³ - βẋ -γẋ³.

To obtain the training data, we solve the equation of motion using one of the
solvers in `OrdinaryDiffEq`:

```julia
using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, LinearAlgebra
k, α, β, γ = 1, 0.1, 0.2, 0.3
tspan = (0.0f0,10.0f0)

function dxdt_train(du,u,p,t)
  du[1] = u[2]
  du[2] = -k*u[1] - α*u[1]^3 - β*u[2] - γ*u[2]^3
end

u0 = [1f0,0f0]
ts = Float32.(collect(0.0:0.1:tspan[2]))
prob_train = ODEProblem{true}(dxdt_train,u0,tspan,p=nothing)
data_train = Array(solve(prob_train,Tsit5(),saveat=ts))
```

Now, we create a TensorLayer that will be able to perform 4th order expansions in
a Legendre Basis:

```julia
A = [LegendreBasis(4), LegendreBasis(4)]
nn = TensorLayer(A, 1)
```

and we also instantiate the model we are trying to learn, "informing" the neural
about the `∝x` and `∝v` dependencies in the equation of motion:

```julia
f = x -> abs(x) > 30 ? sign(x)*30 : x

function dxdt_pred(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]*u[1] - p[2]*u[2] + f(nn(u,p[3:end])[1])
end

θ = zeros(18)
```

Note that we also introduced a "cap" in the neural network term to avoid instabilities
in the solution of the ODE. We also initialized the vector of parameters to zero
in order to obtain a faster convergence for this particular example.

Finally, we introduce the corresponding loss functions:

```julia

function predict_adjoint(θ)
  x = Array(solve(prob_pred,Tsit5(),p=θ,saveat=ts))
end

function loss_adjoint(θ)
  x = predict_adjoint(θ)
  loss = sum(norm.(x - train_data))
  return loss
end

function cb(θ,l)
  @show θ, l
  return false
end
```

and we train the network using two rounds of `ADAM`:

```julia
res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.05), cb = cb,maxiters=200)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.minimizer, ADAM(0.001), cb = cb,maxiters=300)
opt = res2.minimizer
```
