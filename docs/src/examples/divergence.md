# Handling Divergent and Unstable Trajectories

It is not uncommon for a set of parameters in an ODE model to simply give a
divergent trajectory. If the rate of growth compounds and outpaces the rate
of decay, you will end up at infinity in finite time. This it is not uncommon
to see divergent trajectories in the optimization of parameters, as many times
an optimizer can take an excursion into a parameter regime which simply gives
a model with an infinite solution.

This can be addressed by using the retcode system. In DifferentialEquations.jl,
[RetCodes](https://diffeq.sciml.ai/stable/basics/solution/#retcodes) detail
the status of the returned solution. Thus if the retcode corresponds to a
failure, we can use this to give an infinite loss and effectively discard the
parameters. This is shown in the loss function:

```julia
function loss(p)
  tmp_prob = remake(prob, p=p)
  tmp_sol = Array(solve(tmp_prob,Tsit5(),saveat=0.1))
  if size(tmp_sol) == size(dataset)
    return sum(abs2,tmp_sol - dataset)
  else
    return Inf
  end
end
```

A full example making use of this trick is:

```julia
using DifferentialEquations, Plots

function lotka_volterra!(du,u,p,t)
    rab, wol = u
    α,β,γ,δ=p
    du[1] = drab = α*rab - β*rab*wol
    du[2] = dwol = γ*rab*wol - δ*wol
    nothing
end

u0 = [1.0,1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra!,u0,tspan,p)
sol = solve(prob,saveat=0.1)
plot(sol)

dataset = Array(sol)
scatter!(sol.t,dataset')

tmp_prob = remake(prob, p=[1.2,0.8,2.5,0.8])
tmp_sol = solve(tmp_prob)
plot(tmp_sol)
scatter!(sol.t,dataset')

function loss(p)
  tmp_prob = remake(prob, p=p)
  tmp_sol = Array(solve(tmp_prob,Tsit5(),saveat=0.1))
  if size(tmp_sol) == size(dataset)
    return sum(abs2,tmp_sol - dataset)
  else
    return Inf
  end
end

using DiffEqFlux

pinit = [1.2,0.8,2.5,0.8]
res = DiffEqFlux.sciml_train(loss,pinit,ADAM(), maxiters = 1000)

#try Newton method of optimization
res = DiffEqFlux.sciml_train(loss,pinit,Newton(), GalacticOptim.AutoForwardDiff())
```
