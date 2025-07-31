@testsetup module CNFTestSetup

using Reexport

@reexport using Zygote, Distances, Distributions, DistributionsAD, Optimization,
                LinearAlgebra, OrdinaryDiffEq, Random, Test, OptimizationOptimisers,
                Statistics, ComponentArrays

Random.seed!(1999)

function callback(adtype)
    return function (p, l)
        @info "[FFJORD $(nameof(typeof(adtype)))] Loss: $(l)"
        false
    end
end

export callback

end

@testitem "Smoke test for FFJORD" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(1, 1, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(nn, tspan, (1,), Tsit5())
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps)

    data_dist=Beta(2.0f0, 2.0f0)
    train_data=Float32.(rand(data_dist, 1, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    @testset "ADType: $(adtype)" for adtype in (Optimization.AutoForwardDiff(),
        Optimization.AutoReverseDiff(), Optimization.AutoTracker(),
        Optimization.AutoZygote(), Optimization.AutoFiniteDiff())
        @testset "regularize = $(regularize) & monte_carlo = $(monte_carlo)" for regularize in
                                                                                 (
                true, false),
            monte_carlo in (true, false)

            @info "regularize = $(regularize) & monte_carlo = $(monte_carlo)"
            st_ = (; st..., regularize, monte_carlo)
            model = StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)
            optf = Optimization.OptimizationFunction((θ, _) -> loss(model, θ), adtype)
            optprob = Optimization.OptimizationProblem(optf, ps)
            @test !isnothing(Optimization.solve(
                optprob, Adam(0.1); callback = callback(adtype), maxiters = 3)) broken=(adtype isa
                                                                                        Optimization.AutoTracker)
        end
    end
end

@testitem "Smoke test for FFJORDDistribution (sampling & pdf)" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(1, 1, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(nn, tspan, (1,), Tsit5())
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps)

    regularize=false
    monte_carlo=false

    data_dist=Beta(2.0f0, 2.0f0)
    train_data=Float32.(rand(data_dist, 1, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    adtype=Optimization.AutoZygote()

    st_=(; st..., regularize, monte_carlo)
    model=StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)

    optf=Optimization.OptimizationFunction((θ, _)->loss(model, θ), adtype)
    optprob=Optimization.OptimizationProblem(optf, ps)
    res=Optimization.solve(optprob, Adam(0.1); callback = callback(adtype), maxiters = 10)

    ffjord_d=FFJORDDistribution(ffjord_mdl, res.u, st_)

    @test !isnothing(pdf(ffjord_d, train_data))
    @test !isnothing(rand(ffjord_d))
    @test !isnothing(rand(ffjord_d, 10))
end

@testitem "Test for default base distribution and deterministic trace FFJORD" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(1, 1, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(nn, tspan, (1,), Tsit5())
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps)

    regularize=false
    monte_carlo=false

    data_dist=Beta(7.0f0, 7.0f0)
    train_data=Float32.(rand(data_dist, 1, 100))
    test_data=Float32.(rand(data_dist, 1, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    adtype=Optimization.AutoZygote()
    st_=(; st..., regularize, monte_carlo)
    model=StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)

    optf=Optimization.OptimizationFunction((θ, _)->loss(model, θ), adtype)
    optprob=Optimization.OptimizationProblem(optf, ps)
    res=Optimization.solve(optprob, Adam(0.1); callback = callback(adtype), maxiters = 10)

    actual_pdf=pdf.(data_dist, test_data)
    learned_pdf=exp.(model(test_data, res.u)[1])

    @test ps != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.9
end

@testitem "Test for alternative base distribution and deterministic trace FFJORD" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(
        nn, tspan, (1,), Tsit5(); basedist = MvNormal([0.0f0], Diagonal([4.0f0])))
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps)

    regularize=false
    monte_carlo=false

    data_dist=Normal(6.0f0, 0.7f0)
    train_data=Float32.(rand(data_dist, 1, 100))
    test_data=Float32.(rand(data_dist, 1, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    adtype=Optimization.AutoZygote()
    st_=(; st..., regularize, monte_carlo)
    model=StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)

    optf=Optimization.OptimizationFunction((θ, _)->loss(model, θ), adtype)
    optprob=Optimization.OptimizationProblem(optf, ps)
    res=Optimization.solve(optprob, Adam(0.1); callback = callback(adtype), maxiters = 30)

    actual_pdf=pdf.(data_dist, test_data)
    learned_pdf=exp.(model(test_data, res.u)[1])

    @test ps != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
end

@testitem "Test for multivariate distribution and deterministic trace FFJORD" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(2, 2, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(nn, tspan, (2,), Tsit5())
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps)

    regularize=false
    monte_carlo=false

    μ=ones(Float32, 2)
    Σ=Diagonal([7.0f0, 7.0f0])
    data_dist=MvNormal(μ, Σ)
    train_data=Float32.(rand(data_dist, 100))
    test_data=Float32.(rand(data_dist, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    adtype=Optimization.AutoZygote()
    st_=(; st..., regularize, monte_carlo)
    model=StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)

    optf=Optimization.OptimizationFunction((θ, _)->loss(model, θ), adtype)
    optprob=Optimization.OptimizationProblem(optf, ps)
    res=Optimization.solve(
        optprob, Adam(0.01); callback = callback(adtype), maxiters = 30)

    actual_pdf=pdf(data_dist, test_data)
    learned_pdf=exp.(model(test_data, res.u)[1])

    @test ps != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
end

@testitem "Test for multivariate distribution and FFJORD with regularizers" setup=[CNFTestSetup] tags=[:advancedneuralde] begin
    nn=Chain(Dense(2, 2, tanh))
    tspan=(0.0f0, 1.0f0)
    ffjord_mdl=FFJORD(nn, tspan, (2,), Tsit5())
    ps, st=Lux.setup(Xoshiro(0), ffjord_mdl)
    ps=ComponentArray(ps) .* 0.001f0

    regularize=true
    monte_carlo=true

    μ=ones(Float32, 2)
    Σ=Diagonal([7.0f0, 7.0f0])
    data_dist=MvNormal(μ, Σ)
    train_data=Float32.(rand(data_dist, 100))
    test_data=Float32.(rand(data_dist, 100))

    function loss(model, θ)
        logpx, λ₁, λ₂=model(train_data, θ)
        return -mean(logpx)
    end

    adtype=Optimization.AutoZygote()
    st_=(; st..., regularize, monte_carlo)
    model=StatefulLuxLayer{true}(ffjord_mdl, nothing, st_)

    optf=Optimization.OptimizationFunction((θ, _)->loss(model, θ), adtype)
    optprob=Optimization.OptimizationProblem(optf, ps)
    res=Optimization.solve(
        optprob, Adam(0.01); callback = callback(adtype), maxiters = 30)

    actual_pdf=pdf(data_dist, test_data)
    learned_pdf=exp.(model(test_data, res.u)[1])

    @test ps != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
end
