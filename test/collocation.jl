using DiffEqFlux, OrdinaryDiffEq, Test

bounded_support_kernels = [EpanechnikovKernel(), UniformKernel(), TriangularKernel(),
    QuarticKernel(), TriweightKernel(), TricubeKernel(), CosineKernel()]

unbounded_support_kernels = [GaussianKernel(), LogisticKernel(), SigmoidKernel(),
    SilvermanKernel()]

@testset "Kernel Functions" begin
    ts = collect(-5.0:0.1:5.0)
    @testset "Kernels with support from -1 to 1" begin
        minus_one_index = findfirst(x -> ==(x, -1.0), ts)
        plus_one_index = findfirst(x -> ==(x, 1.0), ts)
        @testset "$kernel" for (kernel, x0) in zip(bounded_support_kernels,
            [0.75, 0.50, 1.0, 15.0 / 16.0, 35.0 / 32.0, 70.0 / 81.0, pi / 4.0])
            ws = DiffEqFlux.calckernel.((kernel,), ts)
            # t < -1
            @test all(ws[1:(minus_one_index - 1)] .== 0.0)
            # t > 1
            @test all(ws[(plus_one_index + 1):end] .== 0.0)
            # -1 < t <1
            @test all(ws[(minus_one_index + 1):(plus_one_index - 1)] .> 0.0)
            # t = 0
            @test DiffEqFlux.calckernel(kernel, 0.0) == x0
        end
    end
    @testset "Kernels with unbounded support" begin
        @testset "$kernel" for (kernel, x0) in zip(unbounded_support_kernels,
            [1 / (sqrt(2 * pi)), 0.25, 1 / pi, 1 / (2 * sqrt(2))])
            # t = 0
            @test DiffEqFlux.calckernel(kernel, 0.0) == x0
        end
    end
end

@testset "Collocation of data" begin
    f(u, p, t) = p .* u
    rc = 2
    ps = repeat([-0.001], rc)
    tspan = (0.0, 50.0)
    u0 = 3.4 .+ ones(rc)
    t = collect(range(minimum(tspan), stop = maximum(tspan), length = 1000))
    prob = ODEProblem(f, u0, tspan, ps)
    data = Array(solve(prob, Tsit5(), saveat = t, abstol = 1e-12, reltol = 1e-12))
    @testset "$kernel" for kernel in [
        bounded_support_kernels...,
        unbounded_support_kernels...,
    ]
        u′, u = collocate_data(data, t, kernel, 0.003)
        @test sum(abs2, u - data) < 1e-8
    end
    @testset "$kernel" for kernel in [bounded_support_kernels...]
        # Errors out as the bandwidth is too low
        @test_throws ErrorException collocate_data(data, t, kernel, 0.001)
    end
end
