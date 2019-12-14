using OrdinaryDiffEq, Test

p = [1.5 1.0;3.0 1.0]
function lotka_volterra(du,u,p,t)
  du[1] = p[1,1]*u[1] - p[1,2]*u[1]*u[2]
  du[2] = -p[2,1]*u[2] + p[2,2]*u[1]*u[2]
end

u0 = [1.0,1.0]
tspan = (0.0,10.0)

prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5())

#=
using Plots
plot(sol)
=#

using Flux, DiffEqFlux
p = [2.2 1.0;2.0 0.4] # Tweaked Initial Parameter Array
params = Flux.Params([p])

function predict_adjoint() # Our 1-layer neural network
  diffeq_adjoint(p,prob,Tsit5(),saveat=0.0:0.1:10.0)
end

loss_adjoint() = sum(abs2,x-1 for x in predict_adjoint())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_adjoint())
end

predict_adjoint()

# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_adjoint, params, data, opt, cb = cb)

@test loss_adjoint() < 1
