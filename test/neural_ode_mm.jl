using DiffEqFlux, GalacticOptim, OrdinaryDiffEq, Test

#A desired MWE for now, not a test yet.
function f(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁ + k₃*y₂*y₃
    du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2
    du[3] =  y₁ + y₂ + y₃ - 1
    nothing
end
u₀ = [1.0, 0, 0]
M = [1. 0  0
     0  1. 0
     0  0  0]
tspan = (0.0,1.0)
p = [0.04,3e7,1e4]
func = ODEFunction(f,mass_matrix=M)
prob = ODEProblem(func,u₀,tspan,p)
sol = solve(prob,Rodas5(),saveat=0.1)

dudt2 = FastChain(FastDense(3,64,tanh),FastDense(64,2))
ndae = NeuralODEMM(dudt2, (u,p,t) -> [u[1] + u[2] + u[3] - 1], tspan, M, Rodas5(autodiff=false),saveat=0.1)
ndae(u₀)

function predict_n_dae(p)
    ndae(u₀,p)
end
function loss(p)
    pred = predict_n_dae(p)
    loss = sum(abs2,Array(sol) .- pred)
    loss,pred
end

cb = function (p,l,pred) #callback function to observe training
  display(l)
  return false
end

l1 = first(loss(ndae.p))
optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, ndae.p)
res = GalacticOptim.solve(optprob, BFGS(initial_stepnorm = 0.001), cb = cb, maxiters = 100)
@test res.minimum < l1
