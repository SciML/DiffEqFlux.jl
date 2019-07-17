using Flux, DiffEqFlux, StochasticDiffEq

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
function lotka_volterra_noise(du,u,p,t)
  du[1] = 0.1u[1]
  du[2] = 0.1u[2]
end
prob = SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,10.0))
p = param([2.2, 1.0, 2.0, 0.4])
function predict_fd_sde()
  diffeq_fd(p,sol->sol[1,:],101,prob,SOSRI(),saveat=0.1)
end
loss_fd_sde() = sum(abs2,x-1 for x in predict_fd_sde())
loss_fd_sde()
Flux.back!(loss_fd_sde())

prob = SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,2.0))
function predict_rd_sde()
  Tracker.collect(diffeq_rd(p,prob,SOSRI(),saveat=0.0:0.1:2.0))
end
loss_rd_sde() = sum(abs2,x-1 for x in predict_rd_sde())
loss_rd_sde()
Flux.back!(loss_rd_sde())
