@testitem "Second Order Neural ODE" tags = [:advancedneuralde] begin
    using ComponentArrays, Zygote, Random, Optimization, OptimizationOptimisers,
        OrdinaryDiffEq

    rng = Xoshiro(0)

    u0 = Float32[0.0; 2.0]
    du0 = Float32[0.0; 0.0]
    tspan = (0.0f0, 1.0f0)
    t = range(tspan[1], tspan[2]; length = 20)

    model = Chain(Dense(2, 50, tanh), Dense(50, 2))
    p, st = Lux.setup(rng, model)
    p = ComponentArray(p)
    ff(du, u, p, t) = first(model(u, p, st))
    prob = SecondOrderODEProblem{false}(ff, du0, u0, tspan, p)

    function predict(p)
        return Array(
            solve(
                prob, Tsit5(); p, saveat = t,
                sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP())
            )
        )
    end

    correct_pos = Float32.(
        transpose(
            hcat(
                collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end]
            )
        )
    )

    function loss_n_ode(p)
        pred = predict(p)
        return sum(abs2, correct_pos .- pred[1:2, :])
    end

    l1 = loss_n_ode(p)

    function callback(p, l)
        @info "[SecondOrderODE] Loss: $l"
        return l < 0.01
    end

    optfunc = Optimization.OptimizationFunction(
        (x, p) -> loss_n_ode(x), Optimization.AutoZygote()
    )
    optprob = Optimization.OptimizationProblem(optfunc, p)
    res = Optimization.solve(optprob, Adam(0.01f0); callback = callback, maxiters = 100)
    l2 = loss_n_ode(res.minimizer)
    @test l2 < l1

    function predict(p)
        return Array(
            solve(
                prob, Tsit5(); p, saveat = t,
                sensealg = QuadratureAdjoint(; autojacvec = ZygoteVJP())
            )
        )
    end

    correct_pos = Float32.(
        transpose(
            hcat(
                collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end]
            )
        )
    )

    function loss_n_ode(p)
        pred = predict(p)
        return sum(abs2, correct_pos .- pred[1:2, :])
    end

    optfunc = Optimization.OptimizationFunction(
        (x, p) -> loss_n_ode(x), Optimization.AutoZygote()
    )
    optprob = Optimization.OptimizationProblem(optfunc, p)
    res = Optimization.solve(optprob, Adam(0.01f0); callback = callback, maxiters = 100)
    l2 = loss_n_ode(res.minimizer)
    @test l2 < l1

    function predict(p)
        return Array(
            solve(
                prob, Tsit5(); p, saveat = t,
                sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP())
            )
        )
    end

    correct_pos = Float32.(
        transpose(
            hcat(
                collect(0:0.05:1)[2:end], collect(2:-0.05:1)[2:end]
            )
        )
    )

    function loss_n_ode(p)
        pred = predict(p)
        return sum(abs2, correct_pos .- pred[1:2, :])
    end

    optfunc = Optimization.OptimizationFunction(
        (x, p) -> loss_n_ode(x), Optimization.AutoZygote()
    )
    optprob = Optimization.OptimizationProblem(optfunc, p)
    res = Optimization.solve(optprob, Adam(0.01f0); callback = callback, maxiters = 100)
    l2 = loss_n_ode(res.minimizer)
    @test l2 < l1
end
