using DiffEqFlux, OrdinaryDiffEq, Test

function f(u, p, t)
    p .* u
end

rc = 62
ps = repeat([-0.001], rc)
tspan = (0.0, 50.0)
u0 = 3.4 .+ ones(rc)
t = collect(range(minimum(tspan), stop = maximum(tspan), length = 1000))
prob = ODEProblem(f, u0, tspan, ps)
data = Array(solve(prob, Tsit5(), saveat = t, abstol = 1e-12, reltol = 1e-12))

@testset "$kernel" for kernel in [
    EpanechnikovKernel(),
    UniformKernel(),
    TriangularKernel(),
    QuarticKernel(),
    TriweightKernel(),
    TricubeKernel(),
    CosineKernel(),
    GaussianKernel(),
    LogisticKernel(),
    SigmoidKernel(),
    SilvermanKernel(),
]
    u′, u = collocate_data(data, t, kernel)
    @test sum(abs2, u - data) < 1e-8
end
