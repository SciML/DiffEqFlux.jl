# Precompilation workload for DiffEqFlux
# This improves time-to-first-X (TTFX) by precompiling common code paths

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    # Setup code - imports and minimal test data
    # This code is run during precompilation but the compilation results are discarded
    using Random: MersenneTwister
    using Lux: Chain, Dense

    @compile_workload begin
        # These operations will be precompiled
        # Focus on the most common use cases

        # Use a fixed RNG for reproducibility
        rng = MersenneTwister(0)

        # Create a simple model - this is the most common pattern
        model = Chain(Dense(2, 4, tanh), Dense(4, 2))

        # Create NeuralODE layer - the main entry point
        # Note: We don't run the forward pass because it requires an ODE solver
        # which is not a direct dependency of DiffEqFlux
        tspan = (0.0f0, 1.0f0)
        node = NeuralODE(model, tspan)

        # Setup parameters and state - this is called often and benefits from precompilation
        ps, st = Lux.setup(rng, node)

        # Precompile StatefulLuxLayer creation (used in forward pass)
        stateful = StatefulLuxLayer{true}(node.model, nothing, st)

        # Precompile the dudt function creation pattern
        x0 = Float32[1.0, 0.0]
        dudt_out = stateful(x0, ps)

        # Precompile ODEFunction and ODEProblem creation
        dudt(u, p, t) = stateful(u, p)
        ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
        prob = ODEProblem{false}(ff, x0, node.tspan, ps)

        # Precompile FFJORD constructor
        ffjord_model = Chain(Dense(2, 4, tanh), Dense(4, 2))
        ffjord = FFJORD(ffjord_model, tspan, (2,))

        # Precompile collocation kernel calculations (commonly used)
        tpoints = Float32[0.0, 0.5, 1.0]
        data = Float32[1.0 1.1 1.2; 0.0 0.1 0.2]
        try
            collocate_data(data, tpoints, TriangularKernel())
        catch
            # May fail with small data, but we still get the compilation
        end
    end
end
