using Flux, DiffEqFlux, OrdinaryDiffEq

#A desired MWE for now, not a test yet.

function f(out,du,u,p,t)
    out[1] = - 0.04u[1]              + 1e4*u[2]*u[3] - du[1]
    out[2] = + 0.04u[1] - 3e7*u[2]^2 - 1e4*u[2]*u[3] - du[2]
    out[3] = u[1] + u[2] + u[3] - 1.0
end

u₀ = [1.0, 0, 0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0,10.0)
differential_vars = [true,true,false]
prob = DAEProblem(f,du₀,u₀,tspan,differential_vars=differential_vars)
sol = solve(prob,DABDF2())

M = [1. 0  0
    0  1. 0
    0  0  0]

dudt2 = Chain(x -> x.^3,Dense(2,50,tanh),Dense(50,2))


ndae = NeuralDAE(dudt2, (u,p,t) -> [u[1] + u[2] + u[3] - 1], tspan, Rodas5())

ndae(u₀,M)