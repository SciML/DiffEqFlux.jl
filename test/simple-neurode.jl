using Flux, DiffEqFlux, OrdinaryDiffEq
using Zygote

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

dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
#= ps = Flux.params(dudt) =#

p = DiffEqFlux.destructure(dudt)
ps = Flux.params(p)
dudt_(u,p,t) = DiffEqFlux.restructure(dudt,p)(u)
prob = ODEProblem(dudt_,u0,tspan,p)

diffeq_adjoint(p,u0,prob,Tsit5())

forward(u->diffeq_adjoint(p,u,prob,Tsit5()),u0)

n_ode = x->neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=1e-7,abstol=1e-9)

pred = n_ode(u0) # Get the prediction using the correct initial condition

function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

data = Iterators.repeated((), 10)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_n_ode())
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
