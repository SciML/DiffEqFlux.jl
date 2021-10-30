using DiffEqFlux, GalacticOptim, OrdinaryDiffEq

#A desired MWE for now, not a test yet.

function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁ + k₃*y₂*y₃
  du[2] =  k₁*y₁ - k₃*y₂*y₃ - k₂*y₂^2
  du[3] =  y₁ + y₂ + y₃ - 1
  nothing
end
M = [1. 0  0
     0  1. 0
     0  0  0]
prob_mm = ODEProblem(ODEFunction(rober,mass_matrix=M),[1.0,0.0,0.0],(0.0,10.0),(0.04,3e7,1e4))
sol = solve(prob_mm,Rodas5(),reltol=1e-8,abstol=1e-8)


dudt2 = Chain(x -> x.^3,Dense(6,50,tanh),Dense(50,2))

ndae = NeuralDAE(dudt2, (u,p,t) -> [u[1] + u[2] + u[3] - 1], tspan, M, DImplicitEuler(),
                        differential_vars = [true,true,false])
truedu0 = similar(u₀)
f(truedu0,u₀,p,0.0)

ndae(u₀,truedu0,Float64.(ndae.p))

function predict_n_dae(p)
    ndae(u₀,p)
end

function loss(p)
    pred = predict_n_dae(p)
    loss = sum(abs2,sol .- pred)
    loss,pred
end

p = p .+ rand(3) .* p

optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, p)
res = GalacticOptim.solve(optprob, BFGS(initial_stepnorm = 0.0001))
