using DiffEqFlux, Flux, OrdinaryDiffEq, Plots

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
  display(plot(solve(remake(prob,u0=Flux.data(_x),p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_rd, params(ann,p,_x), data, opt, cb = cb)
