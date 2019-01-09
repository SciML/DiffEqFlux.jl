using DiffEqML, Flux, OrdinaryDiffEq
#using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
prob = ODEProblem(lotka_volterra,[1.0,1.0],(0.0,10.0))

# len = length(range(0.0,stop=10.0,step=0.1)) = 101

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
function predict()
  diffeq_fd(p,vec,101,prob,Tsit5(),saveat=0.1)
end
loss() = sum(abs2,x-1 for x in predict())
loss()

grads = Tracker.gradient(loss, params, nest=true)
grads[p]

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss, params, data, opt, cb = cb)
