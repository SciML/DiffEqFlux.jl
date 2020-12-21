using DiffEqFlux, Optim, OrdinaryDiffEq, DiffEqSensitivity, Test

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

function loss(θ)
  sol = solve(prob,Tsit5(),p=[9.8,θ[1]],callback=cb,sensealg=ForwardDiffSensitivity())
  target = 20.0
  abs2(sol[end][1] - target)
end

res = DiffEqFlux.sciml_train(loss,[0.8],BFGS())
@test loss(res.minimizer) < 1

function loss(θ)
  sol = solve(prob,Tsit5(),p=[9.8,θ[1]],callback=cb,sensealg=ReverseDiffAdjoint())
  target = 20.0
  abs2(sol[end][1] - target)
end

res = DiffEqFlux.sciml_train(loss,[0.8],BFGS())
@test loss(res.minimizer) < 1
