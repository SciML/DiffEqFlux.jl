@testitem "Stiff Nested AD" tags = [:layers] begin
    using ComponentArrays, Zygote, OrdinaryDiffEq, Optimization, OptimizationOptimisers,
        Random

    u0 = [2.0; 0.0]
    datasize = 30
    tspan = (0.0f0, 1.5f0)

    function trueODEfunc(du, u, p, t)
        true_A = [-0.1 2.0; -2.0 -0.1]
        du .= ((u .^ 3)' * true_A)'
        return
    end
    t = range(tspan[1], tspan[2]; length = datasize)
    prob = ODEProblem(trueODEfunc, u0, tspan)
    ode_data = Array(solve(prob, Tsit5(); saveat = t))

    model = Chain(x -> x .^ 3, Dense(2 => 50, tanh), Dense(50 => 2))

    predict_n_ode(lux_model, p) = lux_model(u0, p)
    loss_n_ode(lux_model, p) = sum(abs2, ode_data .- predict_n_ode(lux_model, p))

    function callback(solver)
        return function (p, l)
            @info "[StiffNestedAD $(nameof(typeof(solver)))] Loss: $l"
            return false
        end
    end

    @testset "Solver: $(nameof(typeof(solver)))" for solver in (
            KenCarp4(), Rodas5(), RadauIIA5(),
        )
        neuralde = NeuralODE(model, tspan, solver; saveat = t, reltol = 1.0e-7, abstol = 1.0e-9)
        ps, st = Lux.setup(Xoshiro(0), neuralde)
        ps = ComponentArray(ps)
        lux_model = StatefulLuxLayer{true}(neuralde, ps, st)
        loss1 = loss_n_ode(lux_model, ps)
        optfunc = Optimization.OptimizationFunction(
            (x, p) -> loss_n_ode(lux_model, x), Optimization.AutoZygote()
        )
        optprob = Optimization.OptimizationProblem(optfunc, ps)
        res = Optimization.solve(
            optprob, Adam(0.1); callback = callback(solver), maxiters = 100
        )
        loss2 = loss_n_ode(lux_model, res.u)
        @test loss2 < loss1
    end
end
