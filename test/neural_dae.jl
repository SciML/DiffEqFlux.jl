using Flux, DiffEqFlux, OrdinaryDiffEq

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
tspan = (0.0,10.0)
func = ODEFunction(f,mass_matrix=M)
prob = ODEProblem(f,u₀,tspan,(0.04,3e7,1e4))
sol = solve(prob,Rodas5())


dudt2 = Chain(x -> x.^3,Dense(3,50,tanh),Dense(50,3))


ndae = NeuralDAE(dudt2, (u,p,t) -> [u[1] + u[2] + u[3] - 1], tspan, M, Rodas5())

ndae(u₀)