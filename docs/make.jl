using Documenter, DiffEqFlux

makedocs(
    sitename = "DiffEqFlux.jl",
    authors="Chris Rackauckas et al.",
    clean = true,
    doctest = false,
    modules = [DiffEqFlux],

    format = Documenter.HTML(#analytics = "",
                             assets = ["assets/favicon.ico"],
                             canonical="https://diffeqflux.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "examples/optimization_ode.md",
            "examples/optimization_sde.md",
            "examples/lotka_volterra.md",
            "examples/delay_diffeq.md",
            "examples/neural_ode_sciml.md",
            "examples/neural_ode_flux.md",
            "examples/mnist_neural_ode.md",
            "examples/augmented_neural_ode.md",
            "examples/normalizing_flows.md",
            "examples/local_minima.md",
            "examples/neural_sde.md",
            "examples/pde_constrained.md",
            "examples/optimal_control.md",
            "examples/feedback_control.md",
            "examples/second_order_neural.md",
            "examples/physical_constraints.md",
            "examples/second_order_adjoints.md",
            "examples/jump.md",
            "examples/universal_diffeq.md",
            "examples/minibatch.md",
            "examples/tensor_layer.md"
        ],
        "Basis" => "basis_docs.md",
        "Tensor Product Layer" => "tensor_layer.md",
        "Continuous Normalizing Flows Layer" => "CNFLayer.md",
        "Neural Differential Equation Layers" => "NeuralDELayers.md",
        "Use with Flux Chain and train!" => "Flux.md",
        "FastChain" => "FastChain.md",
        "GPUs" => "GPUs.md",
        "sciml_train" => "Scimltrain.md",
        "Benchmark" => "Benchmark.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
