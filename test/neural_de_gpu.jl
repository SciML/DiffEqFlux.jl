using DiffEqFlux, Lux, LuxCUDA, CUDA, Zygote, OrdinaryDiffEq, StochasticDiffEq, Test,
    Random, ComponentArrays

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
            @testset "kwargs: $(kwargs))" for kwargs in ((; save_everystep = false,
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
                @test !iszero(grads[1])
                @test !iszero(grads[2])

                anode = AugmentedNDELayer(NeuralODE(aug_dudt, tspan, Tsit5(); kwargs...), 2)
                pd, st = Lux.setup(rng, anode)
                pd = ComponentArray(pd) |> gdev
                st = st |> gdev
                grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
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

    aug_diffusion = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(4 => 50, tanh), Flux.Dense(50 => 4))
    else
        Chain(Dense(4 => 50, tanh), Dense(50 => 4))
    end

    tspan = (0.0f0, 0.1f0)
    @testset "NeuralDSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (x, xs),
        solver in (EulerHeun(), LambaEM(), SOSRI())

        sode = NeuralDSDE(dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0,
            dt = 0.01f0)
        pd, st = Lux.setup(rng, sode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])

        sode = NeuralDSDE(aug_dudt, aug_diffusion, tspan, solver;
            saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
        anode = AugmentedNDELayer(sode, 2)
        pd, st = Lux.setup(rng, anode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])
    end

    diffusion_sde = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 4), x -> reshape(x, 2, 2))
    else
        Chain(Dense(2 => 50, tanh), Dense(50 => 4), x -> reshape(x, 2, 2))
    end

    aug_diffusion_sde = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(4 => 50, tanh), Flux.Dense(50 => 16), x -> reshape(x, 4, 4))
    else
        Chain(Dense(4 => 50, tanh), Dense(50 => 16), x -> reshape(x, 4, 4))
    end

    @testset "NeuralSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (x,),
        solver in (EulerHeun(), LambaEM())

        sode = NeuralSDE(dudt, diffusion_sde, tspan, 2, solver;
            saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
        pd, st = Lux.setup(rng, sode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])

        sode = NeuralSDE(aug_dudt, aug_diffusion_sde, tspan, 4, solver;
            saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
        anode = AugmentedNDELayer(sode, 2)
        pd, st = Lux.setup(rng, anode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
        @test !iszero(grads[2][end])
    end

    dudt = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(6 => 50, tanh), Flux.Dense(50 => 2))
    else
        Chain(Dense(6 => 50, tanh), Dense(50 => 2))
    end

    aug_dudt = if nnlib == "Flux"
        Flux.Chain(Flux.Dense(12 => 50, tanh), Flux.Dense(50 => 4))
    else
        Chain(Dense(12 => 50, tanh), Dense(50 => 4))
    end

    @testset "NeuralCDDE u0: $(typeof(u0))" for u0 in (x, xs)
        dode = NeuralCDDE(dudt, (0.0f0, 2.0f0), (u, p, t) -> zero(u), (0.1f0, 0.2f0),
            MethodOfSteps(Tsit5()); saveat = 0.0f0:0.1f0:2.0f0)
        pd, st = Lux.setup(rng, dode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ dode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])

        dode = NeuralCDDE(aug_dudt, (0.0f0, 2.0f0), (u, p, t) -> zero(u), (0.1f0, 0.2f0),
            MethodOfSteps(Tsit5()); saveat = 0.0f0:0.1f0:2.0f0)
        anode = AugmentedNDELayer(dode, 2)
        pd, st = Lux.setup(rng, anode)
        pd = ComponentArray(pd) |> gdev
        st = st |> gdev

        grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
        @test !iszero(grads[1])
        @test !iszero(grads[2])
    end
end
