@testitem "Newton Neural ODE" tags = [:newton] begin
    using ComponentArrays, Zygote, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Random

    Random.seed!(100)

    n = 1 # number of ODEs
    tspan = (0.0f0, 1.0f0)

    d = 5 # number of data pairs
    x = rand(Float32, n, 5)
    y = rand(Float32, n, 5)

    cb = function (p, l)
        @info "[Newton NeuralODE] Loss: $l"
        false
    end

    NN = Chain(Dense(n => 5n, tanh), Dense(5n => n))

    @info "ROCK4"
    nODE = NeuralODE(NN, tspan, ROCK4(); reltol = 1.0f-4, saveat = [tspan[end]])

    ps, st = Lux.setup(Xoshiro(0), nODE)
    ps = ComponentArray(ps)
    stnODE = StatefulLuxLayer{true}(nODE, ps, st)

    # KrylovTrustRegion is hardcoded to use `Array`
    psd, psax = getdata(ps), getaxes(ps)

    loss_function(θ) = sum(abs2, y .- stnODE(x, ComponentArray(θ, psax))[end])
    l1 = loss_function(psd)
    optf = Optimization.OptimizationFunction(
        (x, p) -> loss_function(x), Optimization.AutoZygote()
    )
    optprob = Optimization.OptimizationProblem(optf, psd)

    res = Optimization.solve(optprob, NewtonTrustRegion(); maxiters = 100, callback = cb)
    @test loss_function(res.u) < l1
    res = Optimization.solve(
        optprob, OptimizationOptimJL.Optim.KrylovTrustRegion();
        maxiters = 100, callback = cb
    )
    @test loss_function(res.u) < l1

    @info "ROCK2"
    nODE = NeuralODE(NN, tspan, ROCK2(); reltol = 1.0f-4, saveat = [tspan[end]])
    ps, st = Lux.setup(Xoshiro(0), nODE)
    ps = ComponentArray(ps)
    stnODE = StatefulLuxLayer{true}(nODE, ps, st)

    # KrylovTrustRegion is hardcoded to use `Array`
    psd, psax = getdata(ps), getaxes(ps)

    loss_function(θ) = sum(abs2, y .- stnODE(x, ComponentArray(θ, psax))[end])
    l1 = loss_function(psd)
    optfunc = Optimization.OptimizationFunction(
        (x, p) -> loss_function(x), Optimization.AutoZygote()
    )
    optprob = Optimization.OptimizationProblem(optfunc, psd)

    res = Optimization.solve(optprob, NewtonTrustRegion(); maxiters = 100, callback = cb)
    @test loss_function(res.u) < l1
    res = Optimization.solve(
        optprob, OptimizationOptimJL.Optim.KrylovTrustRegion();
        maxiters = 100, callback = cb
    )
    @test loss_function(res.u) < l1
end
