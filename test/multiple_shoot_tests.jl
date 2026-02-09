@testitem "Multiple Shooting" tags = [:basicneuralde] begin
    using ComponentArrays, Zygote, Optimization, OptimizationOptimisers, OrdinaryDiffEq,
        Test, Random
    using DiffEqFlux: group_ranges

    rng = Xoshiro(0)

    ## Test group partitioning helper function
    @test group_ranges(10, 4) == [1:4, 4:7, 7:10]
    @test group_ranges(10, 5) == [1:5, 5:9, 9:10]
    @test group_ranges(10, 10) == [1:10]
    @test_throws DomainError group_ranges(10, 1)
    @test_throws DomainError group_ranges(10, 11)

    # Test configurations
    test_configs = [
        (
            name = "Vector Test Config",
            u0 = Float32[2.0, 0.0],
            ode_func = (du, u, p, t) -> (du .= ((u .^ 3)' * [-0.1 2.0; -2.0 -0.1])'),
            nn = Chain(x -> x .^ 3, Dense(2 => 16, tanh), Dense(16 => 2)),
            u0s_ensemble = [Float32[2.0, 0.0], Float32[3.0, 1.0]],
        ),
        (
            name = "Multi-D Test Config",
            u0 = Float32[2.0 0.0; 1.0 1.5; 0.5 -1.0],
            ode_func = (
                du, u, p, t,
            ) -> (du .= ((u .^ 3) .* [-0.01 0.02; -0.02 -0.01; 0.01 -0.05])),
            nn = Chain(x -> x .^ 3, Dense(3 => 3, tanh)),
            u0s_ensemble = [
                Float32[2.0 0.0; 1.0 1.5; 0.5 -1.0], Float32[3.0 1.0; 2.0 0.5; 1.5 -0.5],
            ],
        ),
    ]

    for config in test_configs
        @info "Running tests for: $(config.name)"

        ## Define initial conditions and time steps
        datasize = 30
        u0 = config.u0
        tspan = (0.0f0, 5.0f0)
        tsteps = range(tspan[1], tspan[2]; length = datasize)

        # Get the data
        trueODEfunc = config.ode_func
        prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
        ode_data = Array(solve(prob_trueode, Tsit5(); saveat = tsteps))

        # Define the Neural Network
        nn = config.nn
        p_init, st = Lux.setup(rng, nn)
        p_init = ComponentArray(p_init)

        neuralode = NeuralODE(nn, tspan, Tsit5(); saveat = tsteps)
        prob_node = ODEProblem((u, p, t) -> first(nn(u, p, st)), u0, tspan, p_init)

        predict_single_shooting(p) = Array(first(neuralode(u0, p, st)))

        # Define loss function
        loss_function(data, pred) = sum(abs2, data - pred)

        ## Evaluate Single Shooting
        function loss_single_shooting(p)
            pred = predict_single_shooting(p)
            l = loss_function(ode_data, pred)
            return l
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((p, _) -> loss_single_shooting(p), adtype)
        optprob = Optimization.OptimizationProblem(optf, p_init)
        res_single_shooting = Optimization.solve(optprob, Adam(0.05); maxiters = 300)

        loss_ss = loss_single_shooting(res_single_shooting.u)
        @info "Single shooting loss: $(loss_ss)"

        ## Test Multiple Shooting
        group_size = 3
        continuity_term = 200

        function loss_multiple_shooting(p)
            return multiple_shoot(
                p, ode_data, tsteps, prob_node, loss_function, Tsit5(),
                group_size; continuity_term, abstol = 1.0e-8, reltol = 1.0e-6
            )[1] # test solver kwargs
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((p, _) -> loss_multiple_shooting(p), adtype)
        optprob = Optimization.OptimizationProblem(optf, p_init)
        res_ms = Optimization.solve(optprob, Adam(0.05); maxiters = 300)

        # Calculate single shooting loss with parameter from multiple_shoot training
        loss_ms = loss_single_shooting(res_ms.u)
        println("Multiple shooting loss: $(loss_ms)")
        @test loss_ms < 10loss_ss

        # Test with custom loss function
        group_size = 4
        continuity_term = 50

        function continuity_loss_abs2(û_end, u_0)
            return sum(abs2, û_end - u_0) # using abs2 instead of default abs
        end

        function loss_multiple_shooting_abs2(p)
            return multiple_shoot(
                p, ode_data, tsteps, prob_node, loss_function,
                continuity_loss_abs2, Tsit5(), group_size; continuity_term
            )[1]
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction(
            (p, _) -> loss_multiple_shooting_abs2(p), adtype
        )
        optprob = Optimization.OptimizationProblem(optf, p_init)
        res_ms_abs2 = Optimization.solve(optprob, Adam(0.05); maxiters = 300)

        loss_ms_abs2 = loss_single_shooting(res_ms_abs2.u)
        println("Multiple shooting loss with abs2: $(loss_ms_abs2)")
        @test loss_ms_abs2 < loss_ss

        ## Test different SensitivityAlgorithm (default is InterpolatingAdjoint)
        function loss_multiple_shooting_fd(p)
            return multiple_shoot(
                p, ode_data, tsteps, prob_node, loss_function, continuity_loss_abs2,
                Tsit5(), group_size; continuity_term, sensealg = ForwardDiffSensitivity()
            )[1]
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((p, _) -> loss_multiple_shooting_fd(p), adtype)
        optprob = Optimization.OptimizationProblem(optf, p_init)
        res_ms_fd = Optimization.solve(optprob, Adam(0.05); maxiters = 300)

        # Calculate single shooting loss with parameter from multiple_shoot training
        loss_ms_fd = loss_single_shooting(res_ms_fd.u)
        println("Multiple shooting loss with ForwardDiffSensitivity: $(loss_ms_fd)")
        @test loss_ms_fd < 10loss_ss

        # Integration return codes `!= :Success` should return infinite loss.
        # In this case, we trigger `retcode = :MaxIters` by setting the solver option `maxiters=1`.
        loss_fail = multiple_shoot(
            p_init, ode_data, tsteps, prob_node, loss_function,
            Tsit5(), datasize; maxiters = 1, verbose = false
        )[1]
        @test loss_fail == Inf

        ## Test for DomainErrors
        @test_throws DomainError multiple_shoot(
            p_init, ode_data, tsteps, prob_node, loss_function, Tsit5(), 1
        )
        @test_throws DomainError multiple_shoot(
            p_init, ode_data, tsteps, prob_node, loss_function, Tsit5(), datasize + 1
        )

        ## Ensembles
        u0s = config.u0s_ensemble
        function prob_func(prob, i, repeat)
            remake(prob; u0 = u0s[i])
        end
        ensemble_prob = EnsembleProblem(prob_node; prob_func = prob_func)
        ensemble_prob_trueODE = EnsembleProblem(prob_trueode; prob_func = prob_func)
        ensemble_alg = EnsembleThreads()
        trajectories = 2
        ode_data_ensemble = Array(
            solve(
                ensemble_prob_trueODE, Tsit5(), ensemble_alg; trajectories, saveat = tsteps
            )
        )

        group_size = 3
        continuity_term = 200
        function loss_multiple_shooting_ens(p)
            return multiple_shoot(
                p, ode_data_ensemble, tsteps, ensemble_prob, ensemble_alg,
                loss_function, Tsit5(), group_size; continuity_term,
                trajectories, abstol = 1.0e-8, reltol = 1.0e-6
            )[1]
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction(
            (p, _) -> loss_multiple_shooting_ens(p), adtype
        )
        optprob = Optimization.OptimizationProblem(optf, p_init)
        res_ms_ensembles = Optimization.solve(optprob, Adam(0.05); maxiters = 300)

        loss_ms_ensembles = loss_single_shooting(res_ms_ensembles.u)

        println("Multiple shooting loss with EnsembleProblem: $(loss_ms_ensembles)")

        @test loss_ms_ensembles < 10loss_ss
    end
end
