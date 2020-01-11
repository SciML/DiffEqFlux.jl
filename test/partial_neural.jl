using DiffEqFlux, Flux, OrdinaryDiffEq, Test, DiffEqSensitivity

x = Float32[0.8; 0.8]
tspan = (0.0f0,10.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))
p = Float32[-2.0,1.1]
p2,re = Flux.destructure(ann)
_p = [p;p2]

function dudt2_(u,p,t)
    x, y = u
    [(re(p[3:end])(u)[1]),p[1]*y + p[2]*x]
end

prob = ODEProblem(dudt2_,x,tspan,_p)
concrete_solve(prob,Tsit5(),x,_p)

function predict_rd()
  Array(concrete_solve(prob,Tsit5(),x,_p,abstol=1e-7,reltol=1e-5,sensealg=TrackerAdjoint()))
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())
loss_rd()

data = Iterators.repeated((), 300)
opt = Descent(0.0001)
cb = function ()
  println(loss_rd())
  #display(plot(solve(remake(prob,u0=Flux.data(_x),p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

loss1 = loss_rd()
Flux.train!(loss_rd, Flux.params(_p,x), data, opt, cb = cb)
loss2 = loss_rd()
@test 10loss2 < loss1

## Partial Neural Adjoint

u0 = Float32[0.8; 0.8]
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))

p1,re = Flux.destructure(ann)
p2 = Float32[-2.0,1.1]
p3 = [p1;p2]
ps = Flux.params(p3,u0)

function dudt_(du,u,p,t)
    x, y = u
    du[1] = re(p[1:41])(u)[1]
    du[2] = p[end-1]*y + p[end]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
concrete_solve(prob,Tsit5(),u0,p3,abstol=1e-8,reltol=1e-6)

function predict_adjoint()
  Array(concrete_solve(prob,Tsit5(),u0,p3,saveat=0.0:1:25.0))
end
loss_adjoint() = sum(abs2,x-1 for x in predict_adjoint())
loss_adjoint()

data = Iterators.repeated((), 300)
opt = ADAM(0.1)
cb = function ()
  println(loss_adjoint())
  #display(plot(solve(remake(prob,p=Flux.data(p3),u0=Flux.data(u0)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

loss1 = loss_adjoint()
Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
loss2 = loss_adjoint()
@test 10loss2 < loss1
