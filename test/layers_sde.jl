using Flux, DiffEqFlux, StochasticDiffEq, Zygote, Test

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
function lotka_volterra_noise(du,u,p,t)
  du[1] = 0.01u[1]
  du[2] = 0.01u[2]
end
prob = SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,10.0))
p = [2.2, 1.0, 2.0, 0.4]
function predict_fd_sde(p)
  diffeq_fd(p,sol->sol[1,:],101,prob,SOSRI(),saveat=0.1)
end
loss_fd_sde(p) = sum(abs2,x-1 for x in predict_fd_sde(p))
loss_fd_sde(p)

@test !iszero(Zygote.gradient(loss_fd_sde,p)[1])

prob = SDEProblem(lotka_volterra,lotka_volterra_noise,[1.0,1.0],(0.0,0.5))
function predict_rd_sde(p)
  #Array(diffeq_rd(p,prob,SOSRI(),saveat=0.0:0.1:0.5))
  vec(Array(concrete_solve(prob,MethodOfSteps(Tsit5()),prob.u0,p,saveat=0.0:0.1:0.5,reltol=1e-4,sensealg=TrackerAdjoint()))[1,:])
end
loss_rd_sde(p) = sum(abs2,x-1 for x in predict_rd_sde(p))
@test !iszero(Zygote.gradient(loss_rd_sde,p)[1])
