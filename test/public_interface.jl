using DiffEqFlux, Lux, OrdinaryDiffEq, StochasticDiffEq, Test
using Random: Xoshiro

struct TestCollocationKernel <: DiffEqFlux.CollocationKernel end

DiffEqFlux.calckernel(::TestCollocationKernel, t, abs_t) = one(t) - abs_t

function run_layer(layer, input)
    ps, st = Lux.setup(Xoshiro(0), layer)
    return layer(input, ps, st)
end

@testset "NeuralDELayer contract" begin
    layer = NeuralODE(Lux.Dense(1 => 1), (0.0f0, 0.01f0), Tsit5(); saveat = 0.01f0)
    @test layer isa DiffEqFlux.NeuralDELayer
    result = run_layer(layer, Float32[1])
    @test result isa Tuple
    @test length(result) == 2
end

@testset "NeuralSDELayer contract" begin
    layer = NeuralDSDE(
        Lux.Dense(1 => 1), Lux.Dense(1 => 1), (0.0f0, 0.01f0), EulerHeun();
        dt = 0.01f0,
    )
    @test layer isa DiffEqFlux.NeuralSDELayer
    result = run_layer(layer, Float32[1])
    @test result isa Tuple
    @test length(result) == 2
end

@testset "CNFLayer contract" begin
    layer = FFJORD(Lux.Dense(1 => 1), (0.0f0, 0.01f0), (1,), Tsit5())
    @test layer isa DiffEqFlux.CNFLayer
    result = run_layer(layer, reshape(Float32[1], 1, 1))
    @test result isa Tuple
    @test length(result) == 2
end

@testset "CollocationKernel contract" begin
    tpoints = range(0.0, 1.0; length = 12)
    data = reduce(hcat, ([sin(t), cos(t)] for t in tpoints))
    kernels = (
        EpanechnikovKernel(), UniformKernel(), TriangularKernel(), QuarticKernel(),
        TriweightKernel(), TricubeKernel(), GaussianKernel(), CosineKernel(),
        LogisticKernel(), SigmoidKernel(), SilvermanKernel(), TestCollocationKernel(),
    )
    for kernel in kernels
        result = collocate_data(data, tpoints, kernel)
        @test result isa Tuple
        @test size(first(result)) == size(data)
        @test size(last(result)) == size(data)
    end
end
