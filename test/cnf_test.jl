println("Starting precompilation")
using OrdinaryDiffEq
using Flux, DiffEqFlux, GalacticOptim
using Test
using Distributions
using Distances
using Zygote
using DistributionsAD
using LinearAlgebra
using Random
println("Starting tests")

Random.seed!(1999)

## callback to be used by all tests
function cb(p, l)
    @show l
    false
end

# smoke test
@testset "Smoke test for adtype=$adtype & regularize=$rglrz & monte_carlo=$mnt_crl" for
        adtype in [GalacticOptim.AutoForwardDiff(), GalacticOptim.AutoReverseDiff(), GalacticOptim.AutoTracker(),
                   GalacticOptim.AutoZygote(), GalacticOptim.AutoFiniteDiff()],
        rglrz in [false, true],
        mnt_crl in [false, true]
    nn = Chain(
        Dense(1, 1, tanh),
    ) |> f32
    tspan = (0.0f0, 1.0f0)
    ffjord_mdl = FFJORD(nn, tspan, Tsit5())

    data_dist = Beta(2.0f0, 2.0f0)
    train_data = rand(data_dist, 1, 100)

    function loss(θ)
        logpx, λ₁, λ₂ = ffjord_mdl(train_data, θ; regularize=rglrz, monte_carlo=mnt_crl)
        -mean(logpx)
    end

    @test_broken !isnothing(DiffEqFlux.sciml_train(loss, ffjord_mdl.p, ADAM(0.1), adtype; maxiters=10))
end

###
# test for default base distribution and deterministic trace CNF
###

nn = Chain(Dense(1, 1, tanh))
data_dist = Beta(7, 7)
data_train = Float32.(rand(data_dist, 1, 100))
tspan = (0.0f0, 1.0f0)
cnf_test = FFJORD(nn, tspan, Tsit5())

function loss_adjoint(θ)
    logpx = cnf_test(data_train, θ; monte_carlo=false)[1]
    loss = -mean(logpx)
end

optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, cnf_test.p)
res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters=100)

θopt = res.minimizer
data_validate = Float32.(rand(data_dist, 1, 100))
actual_pdf = pdf.(data_dist, data_validate)
# use direct trace calculation for predictions
learned_pdf = exp.(cnf_test(data_validate, θopt; monte_carlo=false)[1])

@test totalvariation(learned_pdf, actual_pdf) / size(data_validate, 2) < 0.35

###
# test for alternative base distribution and deterministic trace CNF
###

nn = Chain(Dense(1, 3, tanh), Dense(3, 1, tanh))
data_dist = Normal(6.0, 0.7)
data_train = Float32.(rand(data_dist, 1, 100))
tspan = (0.0f0, 1.0f0)
cnf_test = FFJORD(nn, tspan, Tsit5(); basedist=MvNormal([0.0f0], Diagonal([4.0f0])))

function loss_adjoint(θ)
    logpx = cnf_test(data_train, θ; monte_carlo=false)[1]
    loss = -mean(logpx)
end


optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, 0.01f0 .* cnf_test.p)
res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters=100)

θopt = res.minimizer
data_validate = Float32.(rand(data_dist, 1, 100))
actual_pdf = pdf.(data_dist, data_validate)
learned_pdf = exp.(cnf_test(data_validate, θopt; monte_carlo=false)[1])

@test totalvariation(learned_pdf, actual_pdf) / size(data_validate, 2) < 0.25

###
# test for multivariate distribution and deterministic trace CNF
###

nn = Chain(Dense(2, 2, tanh))
μ = ones(2)
Σ = Diagonal([7.0, 7.0])
data_dist = MvNormal(μ, Σ)
data_train = Float32.(rand(data_dist, 100))
tspan = (0.0f0, 1.0f0)
cnf_test = FFJORD(nn, tspan, Tsit5())

function loss_adjoint(θ)
    logpx = cnf_test(data_train, θ; monte_carlo=false)[1]
    loss = -mean(logpx)
end

optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, 0.01f0 .* cnf_test.p)
res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters=300)

θopt = res.minimizer
data_validate = Float32.(rand(data_dist, 100))
actual_pdf = pdf(data_dist, data_validate)
learned_pdf = exp.(cnf_test(data_validate, θopt; monte_carlo=false)[1])

@test totalvariation(learned_pdf, actual_pdf) / size(data_validate, 2) < 0.25

###
# test for default multivariate distribution and FFJORD with regularizers
###

nn = Chain(Dense(1, 1, tanh))
data_dist = Beta(7, 7)
data_train = Float32.(rand(data_dist, 1, 100))
tspan = (0.0f0, 1.0f0)
ffjord_test = FFJORD(nn, tspan, Tsit5())

function loss_adjoint(θ)
    logpx, λ₁, λ₂ = ffjord_test(data_train, θ, true)
    return mean(@. -logpx + 0.1 * λ₁ + λ₂)
end

optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_adjoint(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, 0.01f0 .* ffjord_test.p)
res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters=300)

θopt = res.minimizer
data_validate = Float32.(rand(data_dist, 1, 100))
actual_pdf = pdf.(data_dist, data_validate)
learned_pdf = exp.(ffjord_test(data_validate, θopt; monte_carlo=false)[1])

@test totalvariation(learned_pdf, actual_pdf) / size(data_validate, 2) < 0.40
