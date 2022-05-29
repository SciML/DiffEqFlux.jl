using Documenter, DiffEqFlux

makedocs(
    sitename = "DiffEqFlux.jl",
    authors="Chris Rackauckas et al.",
    clean = true,
    doctest = false,
    modules = [DiffEqFlux],

    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://diffeqflux.sciml.ai/stable/"),
    pages=[
        "DiffEqFlux.jl: High Level Scientific Machine Learning (SciML) Pre-Built Architectures" => "index.md",
        "Differential Equation Machine Learning Tutorials" => Any[
            "examples/augmented_neural_ode.md",
            "examples/collocation.md",
            "examples/hamiltonian_nn.md",
            "examples/tensor_layer.md",
            "examples/multiple_shooting.md"
        ],
        "Layer APIs" => Any[
            "Classical Basis Layers" => "layers/BasisLayers.md",
            "Tensor Product Layer" => "layers/TensorLayer.md",
            "Continuous Normalizing Flows Layer" => "layers/CNFLayer.md",
            "Spline Layer" => "layers/SplineLayer.md",
            "Neural Differential Equation Layers" => "layers/NeuralDELayers.md",
            "Hamiltonian Neural Network Layer" => "layers/HamiltonianNN.md"
        ],
        "Utility Function APIs" => Any[
            "Smoothed Collocation" => "utilities/Collocation.md",
            "Multiple Shooting Functionality" => "utilities/MultipleShooting.md",
        ],
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
