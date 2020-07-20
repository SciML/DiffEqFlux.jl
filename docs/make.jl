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
        "DiffEqFlux.jl: Generalized Physics-Informed and Scientific Machine Learning (SciML)" => "index.md",
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
            "examples/collocation.md",
            "examples/pde_constrained.md",
            "examples/optimal_control.md",
            "examples/feedback_control.md",
            "examples/second_order_neural.md",
            "examples/physical_constraints.md",
            "examples/second_order_adjoints.md",
            "examples/jump.md",
            "examples/universal_diffeq.md",
            "examples/minibatch.md",
            "examples/tensor_layer.md",
            "examples/neural_gde.md"
        ],
        "Layers" => Any[
        "Classical Basis Layers" => "layers/BasisLayers.md",
        "Tensor Product Layer" => "layers/TensorLayer.md",
        "Continuous Normalizing Flows Layer" => "layers/CNFLayer.md",
        "Spline Layer" => "layers/SplineLayer.md",
        "Neural Differential Equation Layers" => "layers/NeuralDELayers.md",
        ],
        "Controlling Choices of Adjoints" => "ControllingAdjoints.md",
        "Use with Flux Chain and train!" => "Flux.md",
        "FastChain" => "FastChain.md",
        "Smoothed Collocation" => "Collocation.md",
        "GPUs" => "GPUs.md",
        "sciml_train" => "Scimltrain.md",
        "Benchmark" => "Benchmark.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
