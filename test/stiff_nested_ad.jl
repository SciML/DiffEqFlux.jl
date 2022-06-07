using DiffEqFlux, OrdinaryDiffEq, Test

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

model = Flux.Chain(x -> x.^3,
            Flux.Dense(2,50,tanh),
            Flux.Dense(50,2))
neuralde = NeuralODE(model,tspan,Rodas5(),saveat=t,reltol=1e-7,abstol=1e-9)

function predict_n_ode()
  neuralde(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

data = Iterators.repeated((), 10)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
end

# Display the ODE with the initial parameter values.
cb()

neuralde = NeuralODE(model,tspan,Rodas5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(neuralde)
loss1 = loss_n_ode()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
loss2 = loss_n_ode()
@test loss2 < loss1

neuralde = NeuralODE(model,tspan,KenCarp4(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(neuralde)
loss1 = loss_n_ode()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
loss2 = loss_n_ode()
@test loss2 < loss1

neuralde = NeuralODE(model,tspan,RadauIIA5(),saveat=t,reltol=1e-7,abstol=1e-9)
ps = Flux.params(neuralde)
loss1 = loss_n_ode()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
loss2 = loss_n_ode()
@test loss2 < loss1
