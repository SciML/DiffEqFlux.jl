@testitem "NeuralODE" tags=[:basicneuralde] begin
    using ComponentArrays, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Random
    import Flux

    rng = Xoshiro(0)

    @testset "$(nnlib)" for nnlib in ("Flux", "Lux")
        mp = Float32[0.1, 0.1]
        x = Float32[2.0; 0.0]
        xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]))
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

        @testset "u0: $(typeof(u0))" for u0 in (x, xs)
            @testset "kwargs: $(kwargs))" for kwargs in (
                (; save_everystep = false, save_start = false),
                (; abstol = 1e-12, reltol = 1e-12,
                    save_everystep = false, save_start = false),
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

                anode = AugmentedNDELayer(NeuralODE(aug_dudt, tspan, Tsit5(); kwargs...), 2)
                pd, st = Lux.setup(rng, anode)
                pd = ComponentArray(pd)
                grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
                @test !iszero(grads[1])
                @test !iszero(grads[2])
            end
        end
    end
end

@testitem "NeuralDSDE" tags=[:basicneuralde] begin
    using ComponentArrays, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Random
    import Flux

    rng = Xoshiro(0)

    @testset "$(nnlib)" for nnlib in ("Flux", "Lux")
        mp = Float32[0.1, 0.1]
        x = Float32[2.0; 0.0]
        xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]))
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
        @testset "u0: $(typeof(u0)), solver: $(solver)" for u0 in (x, xs),
            solver in (EulerHeun(), LambaEM(), SOSRI())

            sode = NeuralDSDE(
                dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
            pd, st = Lux.setup(rng, sode)
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])
            @test !iszero(grads[2][end])

            sode = NeuralDSDE(aug_dudt, aug_diffusion, tspan, solver;
                saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
            anode = AugmentedNDELayer(sode, 2)
            pd, st = Lux.setup(rng, anode)
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])
            @test !iszero(grads[2][end])
        end
    end
end

@testitem "NeuralSDE" tags=[:basicneuralde] begin
    using ComponentArrays, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Random
    import Flux

    rng = Xoshiro(0)

    @testset "$(nnlib)" for nnlib in ("Flux", "Lux")
        mp = Float32[0.1, 0.1]
        x = Float32[2.0; 0.0]
        xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]))
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

        diffusion_sde = if nnlib == "Flux"
            Flux.Chain(
                Flux.Dense(2 => 50, tanh), Flux.Dense(50 => 4), x -> reshape(x, 2, 2))
        else
            Chain(Dense(2 => 50, tanh), Dense(50 => 4), x -> reshape(x, 2, 2))
        end

        aug_diffusion_sde = if nnlib == "Flux"
            Flux.Chain(
                Flux.Dense(4 => 50, tanh), Flux.Dense(50 => 16), x -> reshape(x, 4, 4))
        else
            Chain(Dense(4 => 50, tanh), Dense(50 => 16), x -> reshape(x, 4, 4))
        end

        @testset "u0: $(typeof(u0)), solver: $(solver)" for u0 in (x,),
            solver in (EulerHeun(), LambaEM())

            sode = NeuralSDE(dudt, diffusion_sde, tspan, 2, solver;
                saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
            pd, st = Lux.setup(rng, sode)
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ sode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])
            @test !iszero(grads[2][end])

            sode = NeuralSDE(aug_dudt, aug_diffusion_sde, tspan, 4, solver;
                saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
            anode = AugmentedNDELayer(sode, 2)
            pd, st = Lux.setup(rng, anode)
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])
            @test !iszero(grads[2][end])
        end
    end
end

@testitem "NeuralCDDE" tags=[:basicneuralde] begin
    using ComponentArrays, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Random
    import Flux

    rng = Xoshiro(0)

    @testset "$(nnlib)" for nnlib in ("Flux", "Lux")
        mp = Float32[0.1, 0.1]
        x = Float32[2.0; 0.0]
        xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]))
        tspan = (0.0f0, 1.0f0)

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
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ dode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])

            dode = NeuralCDDE(
                aug_dudt, (0.0f0, 2.0f0), (u, p, t) -> zero(u), (0.1f0, 0.2f0),
                MethodOfSteps(Tsit5()); saveat = 0.0f0:0.1f0:2.0f0)
            anode = AugmentedNDELayer(dode, 2)
            pd, st = Lux.setup(rng, anode)
            pd = ComponentArray(pd)

            grads = Zygote.gradient(sum ∘ first ∘ anode, u0, pd, st)
            @test !iszero(grads[1])
            @test !iszero(grads[2])
        end
    end
end

@testitem "DimMover" tags=[:basicneuralde] begin
    using Random

    rng = Xoshiro(0)
    r = rand(2, 3, 4, 5)
    layer = DimMover()
    ps, st = Lux.setup(rng, layer)

    @test first(layer(r, ps, st))[:, :, :, 1] == r[:, :, 1, :]
end

@testset "Neural DE CUDA" tags=[:cuda] skip=:(using LuxCUDA; !LuxCUDA.functional()) begin
    using LuxCUDA, Zygote, OrdinaryDiffEq, StochasticDiffEq, Test, Random, ComponentArrays
    import Flux

    CUDA.allowscalar(false)

    rng = Xoshiro(0)

    const gdev = gpu_device()
    const cdev = cpu_device()

    @testset "Neural DE" begin
        mp = Float32[0.1, 0.1] |> gdev
        x = Float32[2.0; 0.0] |> gdev
        xs = Float32.(hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0])) |> gdev
        tspan = (0.0f0, 1.0f0)

        dudt = Chain(Dense(2 => 50, tanh), Dense(50 => 2))
        aug_dudt = Chain(Dense(4 => 50, tanh), Dense(50 => 4))

        @testset "Neural ODE" begin
            @testset "u0: $(typeof(u0))" for u0 in (x, xs)
                @testset "kwargs: $(kwargs))" for kwargs in (
                    (; save_everystep = false, save_start = false),
                    (; save_everystep = false, save_start = false,
                        sensealg = TrackerAdjoint()),
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
                    broken = hasfield(typeof(kwargs), :sensealg) &&
                             ndims(u0) == 2 &&
                             kwargs.sensealg isa TrackerAdjoint
                    @test begin
                        grads = Zygote.gradient(sum ∘ last ∘ first ∘ node, u0, pd, st)
                        CUDA.@allowscalar begin
                            !iszero(grads[1]) && !iszero(grads[2])
                        end
                    end broken=broken

                    anode = AugmentedNDELayer(
                        NeuralODE(aug_dudt, tspan, Tsit5(); kwargs...), 2)
                    pd, st = Lux.setup(rng, anode)
                    pd = ComponentArray(pd) |> gdev
                    st = st |> gdev
                    @test begin
                        grads = Zygote.gradient(sum ∘ last ∘ first ∘ anode, u0, pd, st)
                        CUDA.@allowscalar begin
                            !iszero(grads[1]) && !iszero(grads[2])
                        end
                    end broken=broken
                end
            end
        end

        diffusion = Chain(Dense(2 => 50, tanh), Dense(50 => 2))
        aug_diffusion = Chain(Dense(4 => 50, tanh), Dense(50 => 4))

        tspan = (0.0f0, 0.1f0)
        @testset "NeuralDSDE u0: $(typeof(u0)), solver: $(solver)" for u0 in (xs,),
            solver in (SOSRI(),)
            # CuVector seems broken on CI but I can't reproduce the failure locally

            sode = NeuralDSDE(
                dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0)
            pd, st = Lux.setup(rng, sode)
            pd = ComponentArray(pd) |> gdev
            st = st |> gdev

            @test_broken begin
                grads = Zygote.gradient(sum ∘ last ∘ first ∘ sode, u0, pd, st)
                CUDA.@allowscalar begin
                    !iszero(grads[1]) && !iszero(grads[2]) && !iszero(grads[2][end])
                end
            end
        end
    end
end
