using DiffEqFlux, Zygote, Distances, Distributions, DistributionsAD, Optimization,
    LinearAlgebra, OrdinaryDiffEq, Random, Test, OptimizationOptimisers, Statistics,
    ComponentArrays

Random.seed!(1999)

## callback to be used by all tests
function callback(adtype)
    return function (p, l)
        @info "[FFJORD $(nameof(typeof(adtype)))] Loss: $(l)"
        false
    end
end

@testset "Smoke test for FFJORD" begin
    nn = Chain(Dense(1, 1, tanh))
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, (1,), Tsit5())
    ps, st = Lux.setup(Random.default_rng(), ffjord_mdl)
    ps = ComponentArray(ps)

    data_dist = Beta(2.0f0, 2.0f0)
    train_data = Float32.(rand(data_dist, 1, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂ = model(train_data, θ)
        return -mean(logpx)
    end

    # FIXME: Tracker
    @testset "ADType: $(adtype)" for adtype in (Optimization.AutoForwardDiff(),
            Optimization.AutoReverseDiff(), # Optimization.AutoTracker(),
            Optimization.AutoZygote(), Optimization.AutoFiniteDiff())

        @testset "regularize = $(regularize) & monte_carlo = $(monte_carlo)" for regularize in (true,
                false), monte_carlo in (true, false)
            @info "regularize = $(regularize) & monte_carlo = $(monte_carlo)"
            st_ = (; st..., regularize, monte_carlo)
            model = Lux.Experimental.StatefulLuxLayer(ffjord_mdl, nothing, st_)
            optf = Optimization.OptimizationFunction((θ, _) -> loss(model, θ), adtype)
            optprob = Optimization.OptimizationProblem(optf, ps)
            @test !isnothing(Optimization.solve(optprob, Adam(0.1);
                callback = callback(adtype), maxiters = 3))
        end
    end
end

# @testset "Smoke test for FFJORDDistribution (sampling & pdf)" begin
#     nn = Chain(Dense(1, 1, tanh))
#     tspan = (0.0f0, 1.0f0)
#     ffjord_mdl = FFJORD(nn, tspan, Tsit5())

#     data_dist = Beta(2.0f0, 2.0f0)
#     train_data = Float32.(rand(data_dist, 1, 100))

#     function loss(θ; regularize, monte_carlo)
#         logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
#         -mean(logpx)
#     end

#     adtype = Optimization.AutoZygote()

#     regularize = false
#     monte_carlo = false

#     optf = Optimization.OptimizationFunction((θ, _) -> loss(θ; regularize, monte_carlo),
#         adtype)
#     optprob = Optimization.OptimizationProblem(optf, ffjord_mdl.p)
#     res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 10)

#     ffjord_d = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p = res.u);
#         regularize,
#         monte_carlo)

#     @test !isnothing(pdf(ffjord_d, train_data))
#     @test !isnothing(rand(ffjord_d))
#     @test !isnothing(rand(ffjord_d, 10))
# end
# @testset "Test for default base distribution and deterministic trace FFJORD" begin
#     nn = Chain(Dense(1, 1, tanh))
#     tspan = (0.0f0, 1.0f0)
#     ffjord_mdl = FFJORD(nn, tspan, Tsit5())
#     regularize = false
#     monte_carlo = false

#     data_dist = Beta(7.0f0, 7.0f0)
#     train_data = Float32.(rand(data_dist, 1, 100))
#     test_data = Float32.(rand(data_dist, 1, 100))

#     function loss(θ)
#         logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
#         -mean(logpx)
#     end

#     adtype = Optimization.AutoZygote()
#     optf = Optimization.OptimizationFunction((θ, _) -> loss(θ), adtype)
#     optprob = Optimization.OptimizationProblem(optf, ffjord_mdl.p)
#     res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 10)

#     actual_pdf = pdf.(data_dist, test_data)
#     learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

#     @test ffjord_mdl.p != res.u
#     @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.9
# end
# @testset "Test for alternative base distribution and deterministic trace FFJORD" begin
#     nn = Chain(Dense(1, 3, tanh),
#         Dense(3, 1, tanh))
#     tspan = (0.0f0, 1.0f0)
#     ffjord_mdl = FFJORD(nn, tspan, Tsit5(); basedist = MvNormal([0.0f0], Diagonal([4.0f0])))
#     regularize = false
#     monte_carlo = false

#     data_dist = Normal(6.0f0, 0.7f0)
#     train_data = Float32.(rand(data_dist, 1, 100))
#     test_data = Float32.(rand(data_dist, 1, 100))

#     function loss(θ)
#         logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
#         -mean(logpx)
#     end

#     adtype = Optimization.AutoZygote()
#     optf = Optimization.OptimizationFunction((θ, _) -> loss(θ), adtype)
#     optprob = Optimization.OptimizationProblem(optf, 0.01f0 * ffjord_mdl.p)
#     res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 300)

#     actual_pdf = pdf.(data_dist, test_data)
#     learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

#     @test 0.01f0 * ffjord_mdl.p != res.u
#     @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
# end
# @testset "Test for multivariate distribution and deterministic trace FFJORD" begin
#     nn = Chain(Dense(2, 2, tanh))
#     tspan = (0.0f0, 1.0f0)
#     ffjord_mdl = FFJORD(nn, tspan, Tsit5())
#     regularize = false
#     monte_carlo = false

#     μ = ones(Float32, 2)
#     Σ = Diagonal([7.0f0, 7.0f0])
#     data_dist = MvNormal(μ, Σ)
#     train_data = Float32.(rand(data_dist, 100))
#     test_data = Float32.(rand(data_dist, 100))

#     function loss(θ)
#         logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
#         -mean(logpx)
#     end

#     adtype = Optimization.AutoZygote()
#     optf = Optimization.OptimizationFunction((θ, _) -> loss(θ), adtype)
#     optprob = Optimization.OptimizationProblem(optf, 0.01f0 * ffjord_mdl.p)
#     res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 300)

#     actual_pdf = pdf(data_dist, test_data)
#     learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

#     @test 0.01f0 * ffjord_mdl.p != res.u
#     @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
# end
# @testset "Test for default multivariate distribution and FFJORD with regularizers" begin
#     nn = Chain(Dense(2, 2, tanh))
#     tspan = (0.0f0, 1.0f0)
#     ffjord_mdl = FFJORD(nn, tspan, Tsit5())
#     regularize = true
#     monte_carlo = true

#     μ = ones(Float32, 2)
#     Σ = Diagonal([7.0f0, 7.0f0])
#     data_dist = MvNormal(μ, Σ)
#     train_data = Float32.(rand(data_dist, 100))
#     test_data = Float32.(rand(data_dist, 100))

#     function loss(θ)
#         logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
#         mean(-logpx .+ 1.0f-1 * λ₁ .+ 1.0f-1 * λ₂)
#     end

#     adtype = Optimization.AutoZygote()
#     optf = Optimization.OptimizationFunction((θ, _) -> loss(θ), adtype)
#     optprob = Optimization.OptimizationProblem(optf, 0.01f0 * ffjord_mdl.p)
#     res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 300)

#     actual_pdf = pdf(data_dist, test_data)
#     learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

#     @test 0.01f0 * ffjord_mdl.p != res.u
#     @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.40
# end
