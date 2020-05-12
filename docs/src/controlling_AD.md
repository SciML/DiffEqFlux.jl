# Controlling Automatic Differentiation

One of the key features of DiffEqFlux.jl is the fact that it has many modes
of differentiation which are available, allowing neural differential equations
and universal differential equations to be fit in the manner that is most
appropriate.

To use the automatic differentiation overloads, the differential equation
just needs to be solved with `concrete_solve`. Thus for example,

```julia
using DiffEqSensitivity, OrdinaryDiffEq, Zygote

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = concrete_solve(prob,Tsit5())
loss(u0,p) = sum(concrete_solve(prob,Tsit5(),u0,p,saveat=0.1))
du0,dp = Zygote.gradient(loss,u0,p)
```

this will compute the gradient of the loss function "sum of the values of the
solution to the ODE at timepoints dt=0.1" using an adjoint method, where `du0`
is the derivative of the loss function with respect to the initial condition
and `dp` is the derivative of the loss function with respect to the parameters.

## Choosing a Differentiation Method

The choice of the method for calculating the gradient is made by passing the
keyword argument `sensealg` to `concrete_solve`. The default choice is dependent
on the type of differential equation and the choice of neural network architecture.

The full listing of differentiation methods is described in the
[DifferentialEquations.jl documentation](https://docs.sciml.ai/latest/analysis/sensitivity/#Sensitivity-Algorithms-1).
That page also has guidelines on how to make the right choice.
