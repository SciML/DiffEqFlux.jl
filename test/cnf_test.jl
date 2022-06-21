using DiffEqFlux, Distances, Distributions, DistributionsAD, Optimization,
    LinearAlgebra, OrdinaryDiffEq, Random, Test

Random.seed!(1999)

## callback to be used by all tests
function callback(p, l)
    @show l
    false
end

@testset "Smoke test for FFJORD" begin
    nn = Flux.Chain(
        Flux.Dense(1, 1, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())

    data_dist = Beta(2.0f0, 2.0f0)
    train_data = rand(data_dist, 1, 100)

    function loss(θ; regularize, monte_carlo)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        -mean(logpx)
    end

    @testset "AutoForwardDiff as adtype" begin
        adtype = Optimization.AutoForwardDiff()

        @testset "regularize=false & monte_carlo=false" begin
            regularize = false
            monte_carlo = false

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=false & monte_carlo=true" begin
            regularize = false
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=false" begin
            regularize = true
            monte_carlo = false

            @test_broken !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=true" begin
            regularize = true
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
    end
    @testset "AutoReverseDiff as adtype" begin
        adtype = Optimization.AutoReverseDiff()

        @testset "regularize=false & monte_carlo=false" begin
            regularize = false
            monte_carlo = false
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=false & monte_carlo=true" begin
            regularize = false
            monte_carlo = true
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=false" begin
            regularize = true
            monte_carlo = false

            @test_broken !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=true" begin
            regularize = true
            monte_carlo = true
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
    end
    @testset "AutoTracker as adtype" begin
        adtype = Optimization.AutoTracker()

        @testset "regularize=false & monte_carlo=false" begin
            regularize = false
            monte_carlo = false
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=false & monte_carlo=true" begin
            regularize = false
            monte_carlo = true
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=false" begin
            regularize = true
            monte_carlo = false

            @test_broken !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=true" begin
            regularize = true
            monte_carlo = true
            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
    end
    @testset "AutoZygote as adtype" begin
        adtype = Optimization.AutoZygote()

        @testset "regularize=false & monte_carlo=false" begin
            regularize = false
            monte_carlo = false

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=false & monte_carlo=true" begin
            regularize = false
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=false" begin
            regularize = true
            monte_carlo = false

            @test_broken !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=true" begin
            regularize = true
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
    end
    @testset "AutoFiniteDiff as adtype" begin
        adtype = Optimization.AutoFiniteDiff()

        @testset "regularize=false & monte_carlo=false" begin
            regularize = false
            monte_carlo = false

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=false & monte_carlo=true" begin
            regularize = false
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=false" begin
            regularize = true
            monte_carlo = false

            @test_broken !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
        @testset "regularize=true & monte_carlo=true" begin
            regularize = true
            monte_carlo = true

            @test !isnothing(DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=10))
        end
    end
end
@testset "Smoke test for FFJORDDistribution (sampling & pdf)" begin
    nn = Flux.Chain(
        Flux.Dense(1, 1, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())

    data_dist = Beta(2.0f0, 2.0f0)
    train_data = rand(data_dist, 1, 100)

    function loss(θ; regularize, monte_carlo)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        -mean(logpx)
    end

    adtype = Optimization.AutoZygote()

    regularize = false
    monte_carlo = false

    res = DiffEqFlux.sciml_train(θ -> loss(θ; regularize, monte_carlo), ffjord_mdl.p, ADAM(1f-1), adtype; callback= callback, maxiters=10)
    ffjord_d = FFJORDDistribution(FFJORD(nn, tspan, Tsit5(); p=res.u); regularize, monte_carlo)

    @test !isnothing(pdf(ffjord_d, train_data))
    @test !isnothing(rand(ffjord_d))
    @test !isnothing(rand(ffjord_d, 10))
end
@testset "Test for default base distribution and deterministic trace FFJORD" begin
    nn = Flux.Chain(
        Flux.Dense(1, 1, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())
    regularize = false
    monte_carlo = false

    data_dist = Beta(7.0f0, 7.0f0)
    train_data = rand(data_dist, 1, 100)
    test_data = rand(data_dist, 1, 100)

    function loss(θ)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        -mean(logpx)
    end

    adtype = Optimization.AutoZygote()
    res = DiffEqFlux.sciml_train(loss, ffjord_mdl.p, ADAM(1f-1), adtype; callback= callback, maxiters=100)

    actual_pdf = pdf.(data_dist, test_data)
    learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

    @test ffjord_mdl.p != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.9
end
@testset "Test for alternative base distribution and deterministic trace FFJORD" begin
    nn = Flux.Chain(
        Flux.Dense(1, 3, tanh),
        Flux.Dense(3, 1, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5(); basedist=MvNormal([0.0f0], Diagonal([4.0f0])))
    regularize = false
    monte_carlo = false

    data_dist = Normal(6.0f0, 0.7f0)
    train_data = rand(data_dist, 1, 100)
    test_data = rand(data_dist, 1, 100)

    function loss(θ)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        -mean(logpx)
    end

    adtype = Optimization.AutoZygote()
    res = DiffEqFlux.sciml_train(loss, 0.01f0 * ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=100)

    actual_pdf = pdf.(data_dist, test_data)
    learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

    @test 0.01f0 * ffjord_mdl.p != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
end
@testset "Test for multivariate distribution and deterministic trace FFJORD" begin
    nn = Flux.Chain(
        Flux.Dense(2, 2, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())
    regularize = false
    monte_carlo = false

    μ = ones(Float32, 2)
    Σ = Diagonal([7.0f0, 7.0f0])
    data_dist = MvNormal(μ, Σ)
    train_data = rand(data_dist, 100)
    test_data = rand(data_dist, 100)

    function loss(θ)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        -mean(logpx)
    end

    adtype = Optimization.AutoZygote()
    res = DiffEqFlux.sciml_train(loss, 0.01f0 * ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=300)

    actual_pdf = pdf(data_dist, test_data)
    learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

    @test 0.01f0 * ffjord_mdl.p != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.25
end
@testset "Test for default multivariate distribution and FFJORD with regularizers" begin
    nn = Flux.Chain(
        Flux.Dense(2, 2, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())
    regularize = true
    monte_carlo = true

    μ = ones(Float32, 2)
    Σ = Diagonal([7.0f0, 7.0f0])
    data_dist = MvNormal(μ, Σ)
    train_data = rand(data_dist, 100)
    test_data = rand(data_dist, 100)

    function loss(θ)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize, monte_carlo)
        mean(-logpx .+ 1f-1 * λ₁ .+ 1f-1 * λ₂)
    end

    adtype = Optimization.AutoZygote()
    res = DiffEqFlux.sciml_train(loss, 0.01f0 * ffjord_mdl.p, ADAM(1f-1), adtype; callback = callback, maxiters=300)

    actual_pdf = pdf(data_dist, test_data)
    learned_pdf = exp.(ffjord_mdl(test_data, res.u; regularize, monte_carlo)[1])

    @test 0.01f0 * ffjord_mdl.p != res.u
    @test totalvariation(learned_pdf, actual_pdf) / size(test_data, 2) < 0.40
end
