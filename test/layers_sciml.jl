using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Optim, Test #using Plots

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
  Array(concrete_solve(prob,Tsit5(),u0,p,saveat=0.1,reltol=1e-4))
end
function loss_rd(p)
  sum(abs2,x-1 for x in predict_rd(p))
end

loss_rd(p)

grads = Zygote.gradient(loss_rd, p)
@test !iszero(grads[1])

cb = function (p,l)
  display(l)
  false
end

# Display the ODE with the current parameter values.
loss1 = loss_rd(p)
pmin = DiffEqFlux.sciml_train(loss_rd, p, ADAM(0.1), cb = cb, maxiters = 100)
loss2 = loss_rd(pmin.minimizer)
@test 10loss2 < loss1

pmin = DiffEqFlux.sciml_train(loss_rd, p, BFGS(initial_stepnorm = 0.01), cb = cb)
loss2 = loss_rd(pmin.minimizer)
@test 10loss2 < loss1

# Forward-mode, R^n -> R^m layer

p = [2.2, 1.0, 2.0, 0.4]
function predict_fd(p)
  vec(Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=0.0:0.1:1.0,reltol=1e-4,sensealg=ForwardDiffSensitivity())))
end
loss_fd(p) = sum(abs2,x-1 for x in predict_fd(p))
loss_fd(p)

grads = Zygote.gradient(loss_fd, p)
@test !iszero(grads[1])

opt = ADAM(0.1)
cb = function (p,l)
  display(l)
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
loss1 = loss_fd(p)
pmin = DiffEqFlux.sciml_train(loss_fd, p, opt, cb = cb, maxiters = 100)
loss2 = loss_fd(pmin.minimizer)
@test 10loss2 < loss1

pmin = DiffEqFlux.sciml_train(loss_fd, p, BFGS(initial_stepnorm = 0.01), cb = cb)
loss2 = loss_fd(pmin.minimizer)
@test 10loss2 < loss1

# Adjoint sensitivity
p = [2.2, 1.0, 2.0, 0.4]
ps = Flux.params(p)
function predict_adjoint(p)
    vec(Array(concrete_solve(prob,Tsit5(),prob.u0,p,saveat=0.1,reltol=1e-4)))
end
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
loss_adjoint(p) = loss_reduction(predict_adjoint(p))
loss_adjoint(p)

grads = Zygote.gradient(loss_adjoint, p)
@test !iszero(grads[1])

opt = ADAM(0.1)
cb = function (p,l)
  display(l)
  #display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
loss1 = loss_adjoint(p)
pmin = DiffEqFlux.sciml_train(loss_adjoint, p, opt, cb = cb, maxiters = 100)
loss2 = loss_adjoint(pmin.minimizer)
@test 10loss2 < loss1

pmin = DiffEqFlux.sciml_train(loss_adjoint, p, BFGS(initial_stepnorm = 0.01), cb = cb)
loss2 = loss_adjoint(pmin.minimizer)
@test 10loss2 < loss1
