@testitem "Neural DAE" tags=[:basicneuralde] begin
    using ComponentArrays, Zygote, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Random

    # A desired MWE for now, not a test yet.

    function rober(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        du[3] = y₁ + y₂ + y₃ - 1
        nothing
    end
    M = [1.0 0 0
         0 1.0 0
         0 0 0]
    prob_mm = ODEProblem(
        ODEFunction(rober; mass_matrix = M), [1.0, 0.0, 0.0], (0.0, 10.0), (0.04, 3e7, 1e4))
    sol = solve(prob_mm, Rodas5(); reltol = 1e-8, abstol = 1e-8)

    dudt2 = Chain(x -> x .^ 3, Dense(6, 50, tanh), Dense(50, 3))

    u₀ = [1.0, 0, 0]
    du₀ = [-0.04, 0.04, 0.0]
    tspan = (0.0, 10.0)

    ndae = NeuralDAE(dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1], tspan,
        DFBDF(); differential_vars = [true, true, false])
    ps, st = Lux.setup(Xoshiro(0), ndae)
    ps = ComponentArray(ps)

    predict_n_dae(p) = first(ndae(u₀, p, st))

    function loss(p)
        pred = predict_n_dae(p)
        loss = sum(abs2, sol .- pred)
        return loss, pred
    end

    @test_broken begin
        optfunc = Optimization.OptimizationFunction(
            (x, p) -> loss(x), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optfunc, ps)
        res = Optimization.solve(optprob, BFGS(; initial_stepnorm = 0.0001))
    end

    # Same stuff with Lux
    rng = Xoshiro(0)
    dudt2 = Chain(x -> x .^ 3, Dense(6, 50, tanh), Dense(50, 2))
    p, st = Lux.setup(rng, dudt2)
    p = ComponentArray(p)
    ndae = NeuralDAE(dudt2, (u, p, t) -> [u[1] + u[2] + u[3] - 1], tspan, M,
        DImplicitEuler(); differential_vars = [true, true, false])
    truedu0 = similar(u₀)

    function predict_n_dae(p)
        ndae(u₀, p, st)[1]
    end

    function loss(p)
        pred = predict_n_dae(p)
        loss = sum(abs2, sol .- pred)
        loss, pred
    end

    @test_broken begin
        optfunc = Optimization.OptimizationFunction(
            (x, p) -> loss(x), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optfunc, p)
        res = Optimization.solve(optprob, BFGS(; initial_stepnorm = 0.0001))
    end
end