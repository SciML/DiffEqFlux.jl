using DiffEqFlux, Flux, OrdinaryDiffEq, Zygote, Test #using Plots
import Tracker

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
prob = ODEProblem(lotka_volterra,[1.0,1.0],(0.0,10.0))
const len = length(range(0.0,stop=10.0,step=0.1)) # 101

# Reverse-mode

p = [2.2, 1.0, 2.0, 0.4]
params = Tracker.Params([Tracker.param(p)])
function predict_rd(p)
  vec(diffeq_rd(p,prob,Tsit5(),saveat=0.1))
end
loss_rd(p) = sum(abs2,x-1 for x in predict_rd(p))
loss_rd()

grads = Tracker.gradient(loss_rd, p)
@test !iszero(grads[1])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_rd()
Flux.train!(loss_rd, Flux.params([p]), data, opt, cb = cb)
loss2 = loss_rd()
@test 10loss2 < loss1

# Forward-mode, R^n -> R^m layer

p = [2.2, 1.0, 2.0, 0.4]
params = Flux.Params([p])
function predict_fd()
  diffeq_fd(p,vec,2*len,prob,Tsit5(),saveat=0.1,abstol=1e-8,reltol=1e-8) # 2 times for 2 output variables
end
loss_fd() = sum(abs2,x-1 for x in predict_fd())
loss_fd()

grads = Zygote.gradient(loss_fd, params)
@test !iszero(grads[p])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_fd())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_fd()
Flux.train!(loss_fd, params, data, opt, cb = cb)
loss2 = loss_fd()
@test 10loss2 < loss1

# Forward-mode, R^n -> R loss

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
function predict_fd2()
  diffeq_fd(p,loss_reduction,nothing,prob,Tsit5(),saveat=0.1,abstol=1e-8,reltol=1e-8)
end
loss_fd2() = predict_fd2()
loss_fd2()

grads = Zygote.gradient(loss_fd2, params)
@test !iszero(grads[p])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_fd2())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_fd2()
Flux.train!(loss_fd2, params, data, opt, cb = cb)
loss2 = loss_fd2()
@test 10loss2 < loss1



# Adjoint sensitivity
p = [2.2, 1.0, 2.0, 0.4]
params = Flux.Params([p])
function predict_adjoint()
    diffeq_adjoint(p,prob,Tsit5())
end
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
loss_adjoint() = loss_reduction(predict_adjoint())
loss_adjoint()

grads = Zygote.gradient(loss_adjoint, params)
@test !iszero(grads[p])

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
loss1 = loss_adjoint()
Flux.train!(loss_adjoint, params, data, opt, cb = cb)
loss2 = loss_adjoint()
@test 10loss2 < loss1
