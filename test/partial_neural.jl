using DiffEqFlux, Flux, OrdinaryDiffEq, Test, Optim, DiffEqSensitivity

x = Float32[0.8; 0.8]
tspan = (0.0f0,10.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))
p = Float32[-2.0,1.1]
p2,re = Flux.destructure(ann)
_p = [p;p2]
θ = [x;_p]

function dudt2_(u,p,t)
    x, y = u
    [(re(p[3:end])(u)[1]),p[1]*y + p[2]*x]
end

prob = ODEProblem(dudt2_,x,tspan,_p)
concrete_solve(prob,Tsit5(),x,_p)

function predict_rd(θ)
  Array(concrete_solve(prob,Tsit5(),θ[1:2],θ[3:end],abstol=1e-7,reltol=1e-5,sensealg=TrackerAdjoint()))
end
loss_rd(p) = sum(abs2,x-1 for x in predict_rd(p))
l = loss_rd(θ)

cb = function (θ,l)
  println(l)
  #display(plot(solve(remake(prob,u0=Flux.data(_x),p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
cb(θ,l)

loss1 = loss_rd(θ)
res = DiffEqFlux.sciml_train(loss_rd, θ, BFGS(initial_stepnorm = 0.01), cb = cb)
loss2 = res.minimum
@test 10loss2 < loss1

## Partial Neural Adjoint

u0 = Float32[0.8; 0.8]
tspan = (0.0f0,25.0f0)

ann = Chain(Dense(2,10,tanh), Dense(10,1))

p1,re = Flux.destructure(ann)
p2 = Float32[-2.0,1.1]
p3 = [p1;p2]
θ = [u0;p3]

function dudt_(du,u,p,t)
    x, y = u
    du[1] = re(p[1:41])(u)[1]
    du[2] = p[end-1]*y + p[end]*x
end
prob = ODEProblem(dudt_,u0,tspan,p3)
concrete_solve(prob,Tsit5(),u0,p3,abstol=1e-8,reltol=1e-6)

function predict_adjoint(θ)
  Array(concrete_solve(prob,Tsit5(),θ[1:2],θ[3:end],saveat=0.0:1:25.0))
end
loss_adjoint(θ) = sum(abs2,x-1 for x in predict_adjoint(θ))
l = loss_adjoint(θ)

cb = function (θ,l)
  println(l)
  #display(plot(solve(remake(prob,p=Flux.data(p3),u0=Flux.data(u0)),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
cb(θ,l)

loss1 = loss_adjoint(θ)
res = DiffEqFlux.sciml_train(loss_adjoint, θ, BFGS(initial_stepnorm = 0.01), cb = cb)
loss2 = res.minimum
@test 10loss2 < loss1
