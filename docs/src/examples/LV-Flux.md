# Lotka-Volterra with Flux.train!

The following is a quick example of optimizing Lotka-Volterra parameters using
the Flux.jl style. As before, we define the ODE that we want to solve:

```julia
using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test #using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [2.2, 1.0, 2.0, 0.4]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
```

Then we define our loss function with `concrete_solve` for the adjoint method:

```julia
function predict_rd(p)
  Array(concrete_solve(prob,Tsit5(),u0,saveat=0.1,reltol=1e-4))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd(p))
```

Now we setup the optimization. Here we choose the `ADAM` optimizer. To tell
Flux what our parameters to optimize are, we use `Flux.params(p)`. To make the
optimizer run for 100 steps we use 100 outputs of blank data, i.e.
`Iterators.repeated((), 100)` (this is where minibatching would go!).

```julia
opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
Flux.train!(loss_rd, Flux.params(p), Iterators.repeated((), 100), opt, cb = cb)
```

And now `p` will be the optimal parameter values for our chosen loss function.
