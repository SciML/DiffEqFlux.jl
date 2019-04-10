using DiffEqFlux, Flux, OrdinaryDiffEq

x = Float32[0.8; 0.8]
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))
p = param(Float32[-2.0,1.1])

function dudt_(u::TrackedArray,p,t)
    x, y = u
    Flux.Tracker.collect([ann(u)[1],p[1]*y + p[2]*x])
end
function dudt_(u::AbstractArray,p,t)
    x, y = u
    [Flux.data(ann(u)[1]),p[1]*y + p[2]*x*y]
end

prob = ODEProblem(dudt_,x,tspan,p)
diffeq_rd(p,prob,Tsit5())
_x = param(x)

function predict_rd()
  Flux.Tracker.collect(diffeq_rd(p,prob,Tsit5(),u0=_x))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())
loss_rd()

data = Iterators.repeated((), 10)
opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,u0=Flux.data(_x),p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_rd, params(ann,p,_x), data, opt, cb = cb)

## Partial Neural Adjoint

u0 = param(Float32[0.8; 0.8])
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))

p1 = Flux.data(DiffEqFlux.destructure(ann))
p2 = Float32[-2.0,1.1]
p3 = param([p1;p2])
ps = Flux.params(p3,u0)

function dudt_(du,u,p,t)
    x, y = u
    du[1] = DiffEqFlux.restructure(ann,p[1:41])(u)[1]
    du[2] = p[end-1]*y + p[end]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
diffeq_adjoint(p3,prob,Tsit5(),u0=u0,abstol=1e-8,reltol=1e-6)

function predict_adjoint()
  diffeq_adjoint(p3,prob,Tsit5(),u0=u0,saveat=0.0:0.1:25.0)
end
loss_adjoint() = sum(abs2,x-1 for x in predict_adjoint())
loss_adjoint()

data = Iterators.repeated((), 10)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=Flux.data(p3),u0=Flux.data(u0)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
