# Neural Ordinary Differential Equations with Flux.train!

The following is the same neural ODE example as before, but now using Flux.jl
directly with `Flux.train!`. Notice that the only difference is that we have to
make the neural network be a `Chain` and use Flux.jl's `Flux.params` implicit
parameter system.

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

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

dudt2 = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
p,re = Flux.destructure(dudt2) # use this p as the initial condition!
dudt(u,p,t) = re(p)(u) # need to restrcture for backprop!
prob = ODEProblem(dudt,u0,tspan)

function predict_n_ode()
  Array(solve(prob,Tsit5(),u0=u0,p=p,saveat=t))
end

function loss_n_ode()
    pred = predict_n_ode()
    loss = sum(abs2,ode_data .- pred)
    loss
end

loss_n_ode() # n_ode.p stores the initial parameters of the neural ODE

cb = function (;doplot=false) #callback function to observe training
  pred = predict_n_ode()
  display(sum(abs2,ode_data .- pred))
  # plot current prediction against data
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,pred[1,:],label="prediction")
  display(plot(pl))
  return false
end

# Display the ODE with the initial parameter values.
cb()

data = Iterators.repeated((), 1000)
Flux.train!(loss_n_ode, Flux.params(u0,p), data, ADAM(0.05), cb = cb)
```

![](https://user-images.githubusercontent.com/1814174/51399500-1f4dd080-1b14-11e9-8c9d-144f93b6eac2.gif)
