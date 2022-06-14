using DiffEqFlux, Lux, BenchmarkTools, OrdinaryDiffEq, Test, Random

rng = Random.default_rng()
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

lux_dudt2 = Lux.Chain(ActivationFunction(x -> x.^3),
             Lux.Dense(2,50,tanh),
             Lux.Dense(50,2))
lux_n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)

lp, st = Lux.setup(rng, lux_dudt2)
lp = Lux.ComponentArray(lp)

function lux_predict_n_ode(p)
  lux_n_ode(u0,p,st)[1]
end

function lux_loss_n_ode(p)
    pred = fast_predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

dudt2 = Flux.Chain((x) -> x.^3,
                Flux.Dense(2,50,tanh),
                Flux.Dense(50,2))
n_ode = NeuralODE(dudt2,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

_,re = Flux.destructure(dudt2)
p = collect(lp)
@test lux_dudt2(ones(2),lp,st)[1] ≈ re(p)(x)
@test lux_loss_n_ode(lp)[1] ≈ loss_n_ode(p)[1]
@test Zygote.gradient((p)->lux_loss_n_ode(p)[1], lp)[1] ≈ Zygote.gradient((p)->loss_n_ode(p)[1], p)[1] rtol=4e-3