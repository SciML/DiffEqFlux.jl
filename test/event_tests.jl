using DiffEqFlux, OrdinaryDiffEq, Test

function f(du,u,p,t)
  du[1] = u[2]
  du[2] = -p[1]
end

function condition(u,t,integrator) # Event when event_f(u,t) == 0
  u[1]
end

function affect!(integrator)
  integrator.u[2] = -integrator.p[2]*integrator.u[2]
end

cb = ContinuousCallback(condition,affect!)
u0 = [50.0,0.0]
tspan = (0.0,15.0)
p = [9.8, 0.8]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5(),callback=cb)

function loss1(θ)
  sol = solve(prob,Tsit5(),p=[9.8,θ[1]],saveat=0.1,callback=cb,sensealg=ForwardDiffSensitivity())
  target = 20.0
  abs2(sol[end][1] - target)
end

function loss2(θ)
  sol = solve(prob,Tsit5(),p=[9.8,θ[1]],saveat=0.1,callback=cb,sensealg=ReverseDiffAdjoint())
  target = 20.0
  abs2(sol[end][1] - target)
end

# Zygote.gradient(loss1,[0.8])

# Zygote.gradient(loss2,[0.8])

# res = DiffEqFlux.sciml_train(loss1,[0.8],BFGS())
# @test loss1(res.minimizer) < 1

# res = DiffEqFlux.sciml_train(loss2,[0.8],BFGS())
# @test loss2(res.minimizer) < 1
