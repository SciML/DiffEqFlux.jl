using ComponentArrays,
    DiffEqFlux, Lux, Zygote, Random, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Test
rng = Random.default_rng()

#A desired MWE for now, not a test yet.
function f(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    nothing
end
u₀ = [1.0, 0, 0]
M = [1.0 0 0
    0 1.0 0
    0 0 0]
tspan = (0.0, 1.0)
p = [0.04, 3e7, 1e4]
func = ODEFunction(f; mass_matrix = M)
prob = ODEProblem(func, u₀, tspan, p)
sol = solve(prob, Rodas5(); saveat = 0.1)

dudt2 = Chain(Dense(3 => 64, tanh), Dense(64 => 2))
p, st = Lux.setup(rng, dudt2)
p = ComponentArray{Float64}(p)
ndae = NeuralODEMM(dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1], tspan, M,
    Rodas5(; autodiff = false); saveat = 0.1)
ndae(u₀, p, st)

function loss(p)
    pred = first(ndae(u₀, p, st))
    loss = sum(abs2, Array(sol) .- pred)
    return loss, pred
end

cb = function (p, l, pred)
    @info "[NeuralODEMM] Loss: $l"
    return false
end

l1 = first(loss(p))
optfunc = Optimization.OptimizationFunction((x, p) -> loss(x), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optfunc, p)
res = Optimization.solve(optprob, BFGS(; initial_stepnorm = 0.001); callback = cb,
    maxiters = 100)
@test res.minimum < l1
