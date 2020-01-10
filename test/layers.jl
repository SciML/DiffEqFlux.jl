using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test #using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [2.2, 1.0, 2.0, 0.4]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)

# Reverse-mode

function predict_rd(p)
  Array(concrete_solve(prob,Tsit5(),u0,p,saveat=0.1,reltol=1e-4,sensealg=TrackerAdjoint()))
end
loss_rd(p) = sum(abs2,x-1 for x in predict_rd(p))
loss_rd() = sum(abs2,x-1 for x in predict_rd(p))
loss_rd()

grads = Zygote.gradient(loss_rd, p)
@test !iszero(grads[1])

opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_rd()
Flux.train!(loss_rd, Flux.params(p), Iterators.repeated((), 100), opt, cb = cb)
loss2 = loss_rd()
@test 10loss2 < loss1

# Forward-mode, R^n -> R^m layer

p = [2.2, 1.0, 2.0, 0.4]
function predict_fd()
  vec(Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=0.1,reltol=1e-4,sensealg=ForwardDiffSensitivity())))
end
loss_fd() = sum(abs2,x-1 for x in predict_fd())
loss_fd()

ps = Flux.params(p)
grads = Zygote.gradient(loss_fd, ps)
@test !iszero(grads[p])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_fd())
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_fd()
Flux.train!(loss_fd, ps, data, opt, cb = cb)
loss2 = loss_fd()
@test 10loss2 < loss1

# Adjoint sensitivity
p = [2.2, 1.0, 2.0, 0.4]
ps = Flux.params(p)
function predict_adjoint()
    vec(Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=0.1,reltol=1e-4)))
end
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
loss_adjoint() = loss_reduction(predict_adjoint())
loss_adjoint()

grads = Zygote.gradient(loss_adjoint, ps)
@test !iszero(grads[p])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_adjoint()
Flux.train!(loss_adjoint, ps, data, opt, cb = cb)
loss2 = loss_adjoint()
@test 10loss2 < loss1
