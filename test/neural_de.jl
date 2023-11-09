using ComponentArrays,
    DiffEqFlux, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random
import Flux

rng = Random.default_rng()

@testset "Neural DE: $(nnlib)" for nnlib in ("Flux", "Lux")
    mp = Float32[0.1, 0.1]
    x = Float32[2.0; 0.0]
    xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]))
    tspan = (0.0f0, 1.0f0)

    dudt = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 2))
    end

    @testset "Neural ODE" begin
        @testset "u0: $(typeof(u0))" for u0 in (x, xs)
            @testset "kwargs: $(kwargs))" for kwargs in ((; save_everystep = false,
                    save_start = false),
                (; abstol = 1e-12, reltol = 1e-12, save_everystep = false,
                    save_start = false),
                (; save_everystep = false, save_start = false, sensealg = TrackerAdjoint()),
                (; save_everystep = false, save_start = false,
                    sensealg = BacksolveAdjoint()),
                (; saveat = 0.0f0:0.1f0:1.0f0),
                (; saveat = 0.1f0),
                (; saveat = 0.0f0:0.1f0:1.0f0, sensealg = TrackerAdjoint()),
                (; saveat = 0.1f0, sensealg = TrackerAdjoint()))
                node = NeuralODE(dudt, tspan, Tsit5(); kwargs...)
                pd, st = Lux.setup(rng, node)
                pd = ComponentArray(pd)
                grads = Zygote.gradient(sum ∘ first ∘ node, u0, pd, st)
                @test !iszero(grads[1])
                @test !iszero(grads[2])
            end
        end
    end

    diffusion = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 2))
    end

    tspan = (0.0f0, 0.1f0)
    @testset "NeuralDSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (x, xs),
        solver in (EulerHeun(), LambaEM(), SOSRI())

        sode = NeuralDSDE(dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0,
            dt = 0.01f0)
        pd, st = Lux.setup(rng, sode)
        pd = ComponentArray(pd)

        grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])
    end

    diffusion_sde = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 4), x -> reshape(x, 2, 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 4), x -> reshape(x, 2, 2))
    end

    @testset "NeuralSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (x,),
        solver in (EulerHeun(), LambaEM())

        sode = NeuralSDE(dudt, diffusion_sde, tspan, 2, solver;
            saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
        pd, st = Lux.setup(rng, sode)
        pd = ComponentArray(pd)

        grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])
    end
end



# ddudt = Flux.Chain(Flux.Dense(6, 50, tanh), Flux.Dense(50, 2))
dudt = if nnlib == "Flux"
    Flux.Chain(Flux.Dense(6 => 50, tanh), Flux.Dense(50 => 2))
else
    Chain(Dense(6 => 50, tanh), Dense(50 => 2))
end
dode = NeuralCDDE(dudt, (0.0f0, 2.0f0), (p, t) -> zero(x), (0.1f0, 0.2f0),
    MethodOfSteps(Tsit5()); saveat = 0.0:0.1:2.0)
pd, st = Lux.setup(rng, dode)
pd = ComponentArray(pd)

dode(xs, pd, st)

# grads = Zygote.gradient(() -> sum(dode(x)), Flux.params(x, dode))
# @test !iszero(grads[x])
# @test !iszero(grads[dode.p])

# @test_broken grads = Zygote.gradient(() -> sum(dode(xs)), Flux.params(xs, dode)) isa Tuple
# @test_broken !iszero(grads[xs])
# @test !iszero(grads[dode.p])

# @testset "DimMover" begin
#     r = rand(2, 3, 4, 5)
#     @test r[:, :, 1, :] == FluxBatchOrder(r)[:, :, :, 1]
# end
