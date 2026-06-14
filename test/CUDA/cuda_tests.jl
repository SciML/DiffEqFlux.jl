using DiffEqFlux, Lux, LuxCUDA, Zygote, OrdinaryDiffEq, StochasticDiffEq, Test, Random,
    ComponentArrays
import Flux

if !LuxCUDA.functional()
    @info "CUDA not functional, skipping CUDA tests"
    @testset "Neural DE CUDA" begin
        @test_broken false
    end
else
    CUDA.allowscalar(false)

    rng = Xoshiro(0)

    const gdev = gpu_device()
    const cdev = cpu_device()

    @testset "Neural DE CUDA" begin
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
                        (;
                            save_everystep = false, save_start = false,
                            sensealg = TrackerAdjoint(),
                        ),
                        (;
                            save_everystep = false, save_start = false,
                            sensealg = BacksolveAdjoint(),
                        ),
                        (; saveat = 0.0f0:0.1f0:1.0f0),
                        (; saveat = 0.1f0),
                        (; saveat = 0.0f0:0.1f0:1.0f0, sensealg = TrackerAdjoint()),
                        (; saveat = 0.1f0, sensealg = TrackerAdjoint()),
                    )
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
                    end broken = broken

                    anode = AugmentedNDELayer(
                        NeuralODE(aug_dudt, tspan, Tsit5(); kwargs...), 2
                    )
                    pd, st = Lux.setup(rng, anode)
                    pd = ComponentArray(pd) |> gdev
                    st = st |> gdev
                    @test begin
                        grads = Zygote.gradient(sum ∘ last ∘ first ∘ anode, u0, pd, st)
                        CUDA.@allowscalar begin
                            !iszero(grads[1]) && !iszero(grads[2])
                        end
                    end broken = broken
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
                dudt, diffusion, tspan, solver; saveat = 0.0f0:0.01f0:0.1f0, dt = 0.01f0
            )
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
