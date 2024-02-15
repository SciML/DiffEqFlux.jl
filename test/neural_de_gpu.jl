using DiffEqFlux, Lux, LuxCUDA, CUDA, Zygote, OrdinaryDiffEq, StochasticDiffEq, Test,
      Random, ComponentArrays
import Flux

CUDA.allowscalar(false)

rng = Random.default_rng()

const gdev = gpu_device()
const cdev = cpu_device()

@testset "[CUDA] Neural DE: $(nnlib)" for nnlib in ("Flux", "Lux")
    mp = Float32[0.1, 0.1] |> gdev
    x = Float32[2.0; 0.0] |> gdev
    xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0])) |> gdev
    tspan = (0.0f0, 1.0f0)

    dudt = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 2))
    end

    aug_dudt = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(4 => 50, tanh), Flux.Dense(50 => 4))
    else
        Chain(Dense(4 => 50, tanh), Dense(50 => 4))
    end

    @testset "Neural ODE" begin
        @testset "u0: $(typeof(u0))" for u0 in (x, xs)
            @testset "kwargs: $(kwargs))" for kwargs in (
                (; save_everystep = false,
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
                pd = ComponentArray(pd) |> gdev
                st = st |> gdev
                grads = Zygote.gradient(sum ∘ first ∘ node, u0, pd, st)
                CUDA.@allowscalar begin
                    @test !iszero(grads[1])
                    @test !iszero(grads[2])
                end

                anode = AugmentedNDELayer(NeuralODE(aug_dudt, tspan, Tsit5(); kwargs...), 2)
                pd, st = Lux.setup(rng, anode)
                pd = ComponentArray(pd) |> gdev
                st = st |> gdev
                grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
                CUDA.@allowscalar begin
                    @test !iszero(grads[1])
                    @test !iszero(grads[2])
                end
            end
        end
    end

    diffusion = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 2))
    end

    aug_diffusion = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(4 => 50, tanh), Flux.Dense(50 => 4))
    else
        Chain(Dense(4 => 50, tanh), Dense(50 => 4))
    end

    tspan = (0.0f0, 0.1f0)
    @testset "NeuralDSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (xs,),
        solver in (SOSRI(),)
        # CuVector seems broken on CI but I can't reproduce the failure locally

        sode = NeuralDSDE(dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0,
            dt = 0.01f0)
        pd, st = Lux.setup(rng, sode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
        CUDA.@allowscalar begin
            @test !iszero(grads[1])
            @test !iszero(grads[2])
            @test !iszero(grads[2][end])
        end
    end
end
