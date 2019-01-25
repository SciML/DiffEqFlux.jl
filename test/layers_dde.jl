using Flux, DiffEqFlux, DelayDiffEq, Plots

## Setup DDE to optimize
function delay_lotka_volterra(du,u,h,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
  du[2] = dy = (δ*x - γ)*y
end
h(p,t) = ones(eltype(p),2)
prob = DDEProblem(delay_lotka_volterra,[1.0,1.0],h,(0.0,10.0),constant_lags=[0.1])
p = param([2.2, 1.0, 2.0, 0.4])
function predict_fd_dde()
  diffeq_fd(p,sol->sol[1,:],101,prob,MethodOfSteps(Tsit5()),saveat=0.1)
end
loss_fd_dde() = sum(abs2,x-1 for x in predict_fd_dde())
@test_broken loss_fd_dde()
@test_broken Flux.back!(loss_fd_dde())

function predict_rd_dde()
  diffeq_rd(p,prob,MethodOfSteps(Tsit5()),saveat=0.1)[1,:]
end
loss_rd_dde() = sum(abs2,x-1 for x in predict_rd_dde())
loss_rd_dde()
Flux.back!(loss_rd_dde())
