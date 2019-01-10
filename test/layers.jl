using DiffEqFlux, Flux, OrdinaryDiffEq, Test
#using Plots

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
prob = ODEProblem(lotka_volterra,[1.0,1.0],(0.0,10.0))
const len = length(range(0.0,stop=10.0,step=0.1)) # 101

# Reverse-mode

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
function predict_rd()
  diffeq_rd(p,vec,prob,Tsit5(),saveat=0.1)
end
loss_rd() = sum(abs2,x-1 for x in predict_rd())
loss_rd()

grads = Tracker.gradient(loss_rd, params, nest=true)
grads[p]

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_rd())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)

# Forward-mode, R^n -> R^m layer

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
function predict_fd()
  diffeq_fd(p,vec,2*len,prob,Tsit5(),saveat=0.1) # 2 times for 2 output variables
end
loss_fd() = sum(abs2,x-1 for x in predict_fd())
loss_fd()

@test_broken begin
  grads = Tracker.gradient(loss_fd, params, nest=true)
  grads[p]
end

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_fd())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_fd, params, data, opt, cb = cb)

# Forward-mode, R^n -> R loss

p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
loss_reduction(sol) = sum(abs2,x-1 for x in vec(sol))
function predict_fd2()
  diffeq_fd(p,loss_reduction,nothing,prob,Tsit5(),saveat=0.1)
end
loss_fd2() = predict_fd2()
loss_fd2()

grads = Tracker.gradient(loss_fd2, params, nest=true)
grads[p]

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_fd2())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_fd2, params, data, opt, cb = cb)

# Adjoint sensitivity
p = param([2.2, 1.0, 2.0, 0.4])
params = Flux.Params([p])
function predict_adjoint()
    diffeq_adjoint(p,prob,Tsit5())
end
loss_adjoint() = loss_reduction(predict_adjoint())
loss_adjoint()

grads = Tracker.gradient(loss_adjoint, params, nest=true)
grads[p]

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function ()
  display(loss_adjoint())
  #display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the current parameter values.
cb()

Flux.train!(loss_adjoint, params, data, opt, cb = cb)
