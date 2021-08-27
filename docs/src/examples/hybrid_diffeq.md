# Training Neural Networks in Hybrid Differential Equations

Hybrid differential equations are differential equations with implicit or
explicit discontinuities as specified by [callbacks](https://diffeq.sciml.ai/stable/features/callback_functions/).
In the following example, explicit dosing times are given for a pharmacometric
model and the universal differential equation is trained to uncover the missing
dynamical equations.

```julia
using DiffEqFlux, DifferentialEquations, Plots
u0 = Float32[2.; 0.]
datasize = 100
tspan = (0.0f0,10.5f0)
dosetimes = [1.0,2.0,4.0,8.0]

function affect!(integrator)
    integrator.u = integrator.u.+1
end
cb_ = PresetTimeCallback(dosetimes,affect!,save_positions=(false,false))
function trueODEfunc(du,u,p,t)
    du .= -u
end
t = range(tspan[1],tspan[2],length=datasize)

prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),callback=cb_,saveat=t))
dudt2 = Chain(Dense(2,50,tanh),
             Dense(50,2))
p,re = Flux.destructure(dudt2) # use this p as the initial condition!

function dudt(du,u,p,t)
    du[1:2] .= -u[1:2]
    du[3:end] .= re(p)(u[1:2]) #re(p)(u[3:end])
end
z0 = Float32[u0;u0]
prob = ODEProblem(dudt,z0,tspan)

affect!(integrator) = integrator.u[1:2] .= integrator.u[3:end]
cb = PresetTimeCallback(dosetimes,affect!,save_positions=(false,false))

function predict_n_ode()
    _prob = remake(prob,p=p)
    Array(solve(_prob,Tsit5(),u0=z0,p=p,callback=cb,saveat=t,sensealg=ReverseDiffAdjoint()))[1:2,:]
    #Array(solve(prob,Tsit5(),u0=z0,p=p,saveat=t))[1:2,:]
end

function loss_n_ode()
    pred = predict_n_ode()
    loss = sum(abs2,ode_data .- pred)
    loss
end
loss_n_ode() # n_ode.p stores the initial parameters of the neural ODE

cba = function (;doplot=false) #callback function to observe training
  pred = predict_n_ode()
  display(sum(abs2,ode_data .- pred))
  # plot current prediction against data
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,pred[1,:],label="prediction")
  display(plot(pl))
  return false
end
cba()

ps = Flux.params(p)
data = Iterators.repeated((), 200)
Flux.train!(loss_n_ode, ps, data, ADAM(0.05), cb = cba)
```

![Hybrid Universal Differential Equation](https://user-images.githubusercontent.com/1814174/91687561-08fc5900-eb2e-11ea-9f26-6b794e1e1248.gif)

## Note on Sensitivity Methods

The continuous adjoint sensitivities `BacksolveAdjoint`, `InterpolatingAdjoint`,
and `QuadratureAdjoint` are compatible with events for ODEs. `BacksolveAdjoint` and
`InterpolatingAdjoint` can also handle events for SDEs. Use `BacksolveAdjoint` if
the event terminates the time evolution and several states are saved. Currently,
the continuous adjoint sensitivities do not support multiple events per time point.

All methods based on discrete sensitivity analysis via automatic differentiation,
like `ReverseDiffAdjoint`,m`TrackerAdjoint`, or `ForwardDiffAdjoint` are the methods
to use (and `ReverseDiffAdjoint` is demonstrated above), are compatible with events.
This applies to SDEs, DAEs, and DDEs as well.
