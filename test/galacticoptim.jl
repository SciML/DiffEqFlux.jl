using DiffEqFlux, GalacticOptim, GalacticNLopt, OrdinaryDiffEq, Test # , Plots

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
  Array(solve(prob,Tsit5(),p=p,saveat=0.1,reltol=1e-4))
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

optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_rd(x), GalacticOptim.AutoForwardDiff())
optprob = GalacticOptim.OptimizationProblem(optfunc, p)
res = GalacticOptim.solve(optprob, ADAM(0.1), callback = cb, maxiters = 100)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, BFGS(initial_stepnorm = 0.01), callback = cb, allow_f_increases=false)
@test 10res.minimum < loss1

optprob = GalacticOptim.OptimizationProblem(optfunc, p, lb = [0.0 for i in 1:4], ub = [5.0 for i in 1:4])
res = GalacticOptim.solve(optprob, Fminbox(BFGS(initial_stepnorm = 0.01)), callback = cb, allow_f_increases=false)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, Opt(:LN_BOBYQA, 4), maxiters=100, callback = cb)
@test res.minimum < loss1

# Forward-mode, R^n -> R^m layer

p = [2.2, 1.0, 2.0, 0.4]
function predict_fd(p)
  vec(Array(solve(prob,Tsit5(),p=p,saveat=0.0:0.1:1.0,reltol=1e-4,sensealg=ForwardDiffSensitivity())))
end
loss_fd(p) = sum(abs2,x-1 for x in predict_fd(p))
loss_fd(p)

grads = Zygote.gradient(loss_fd, p)
@test !iszero(grads[1])

opt = ADAM(0.1)
cb = function (p,l)
  display(l)
  # display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
loss1 = loss_fd(p)
optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_fd(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, p)

res = GalacticOptim.solve(optprob, opt, callback = cb, maxiters = 100)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, BFGS(initial_stepnorm = 0.01), callback = cb)
@test 10res.minimum < loss1

# Adjoint sensitivity
p = [2.2, 1.0, 2.0, 0.4]
ps = Flux.params(p)
function predict_adjoint(p)
    vec(Array(solve(prob,Tsit5(),u0=eltype(p).(prob.u0),p=p,saveat=0.1,reltol=1e-4)))
end
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
loss_adjoint(p) = loss_reduction(predict_adjoint(p))
loss_adjoint(p)

grads = Zygote.gradient(loss_adjoint, p)
@test !iszero(grads[1])

opt = ADAM(0.1)
cb = function (p,l)
  display(l)
  # display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
  false
end

# Display the ODE with the current parameter values.
loss1 = loss_adjoint(p)
optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoFiniteDiff())
optprob = GalacticOptim.OptimizationProblem(optfunc, p)

res = GalacticOptim.solve(optprob, opt, callback = cb, maxiters = 100)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, BFGS(initial_stepnorm = 0.01), callback = cb)
@test 10res.minimum < loss1

optprob = GalacticOptim.OptimizationProblem(optfunc, p, lb = [0.0 for i in 1:4], ub = [5.0 for i in 1:4])
res = GalacticOptim.solve(optprob, Fminbox(BFGS(initial_stepnorm = 0.01)), callback = cb, maxiters = 100, time_limit = 5, f_calls_limit = 100)
@test 10res.minimum < loss1

opt = Opt(:LD_MMA, 4)
res = GalacticOptim.solve(optprob, opt, maxiters = 100)
@test 10res.minimum < loss1

function lotka_volterra2(u,p,t)
  x, y = u
  α, β, δ, γ = p
  dx = (α - β*y)x
  dy = (δ*x - γ)y
  [dx,dy]
end
p = [2.2, 1.0, 2.0, 0.4]
u0 = [1.0,1.0]
prob = ODEProblem{false}(lotka_volterra2,u0,(0.0,10.0),p)
function predict_adjoint(p)
    vec(Array(solve(prob,Tsit5(),p=p,saveat=0.1,reltol=1e-4,sensealg=InterpolatingAdjoint())))
end
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
loss_adjoint(p) = loss_reduction(predict_adjoint(p))
optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, p)


res = GalacticOptim.solve(optprob, Newton(), maxiters = 100)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, NewtonTrustRegion(), maxiters = 100)
@test 10res.minimum < loss1

res = GalacticOptim.solve(optprob, Optim.KrylovTrustRegion(), maxiters = 100)
@test 10res.minimum < loss1
