using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    # Setup code - import only what's needed
    using Random: Xoshiro
    using Lux: Chain, Dense

    @compile_workload begin
        # Precompile NeuralODE construction with Lux layers
        # This covers the most common use case
        rng = Xoshiro(0)

        # Simple model - 2D input/output with one hidden layer
        model = Chain(Dense(2 => 8, tanh), Dense(8 => 2))
        tspan = (0.0f0, 1.0f0)

        # Create NeuralODE - this is the main entry point
        node = NeuralODE(model, tspan)

        # Setup parameters - users almost always call this
        ps, st = Lux.setup(rng, node)

        # Precompile problem construction (without solving)
        # This happens inside the NeuralODE call
        x = Float32[1.0, 0.0]
        smodel = StatefulLuxLayer{true}(model, nothing, st)
        dudt(u, p, t) = smodel(u, p)
        ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
        prob = ODEProblem{false}(ff, x, tspan, ps)

        # Precompile AugmentedNDELayer path
        aug_model = Chain(Dense(4 => 8, tanh), Dense(8 => 4))
        aug_node = NeuralODE(aug_model, tspan)
        anode = AugmentedNDELayer(aug_node, 2)
        ps_aug, st_aug = Lux.setup(rng, anode)

        # Precompile DimMover
        dm = DimMover()
        ps_dm, st_dm = Lux.setup(rng, dm)

        # Precompile FFJORD construction
        ffjord_model = Chain(Dense(2 => 8, tanh), Dense(8 => 2))
        ffjord = FFJORD(ffjord_model, tspan, (2,))
        ps_ffjord, st_ffjord = Lux.setup(rng, ffjord)
    end
end
