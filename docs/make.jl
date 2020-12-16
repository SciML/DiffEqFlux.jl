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
        "Basic Parameter Fitting Tutorials" => Any[
            "examples/optimization_ode.md",
            "examples/optimization_sde.md",
            "examples/lotka_volterra.md",
            "examples/delay_diffeq.md",
            "examples/pde_constrained.md",
            ],
        "Neural ODE and SDE Tutorials" => Any[
            "examples/neural_ode_sciml.md",
            "examples/neural_ode_flux.md",
            "examples/mnist_neural_ode.md",
            "examples/neural_sde.md",
            "examples/augmented_neural_ode.md",
            "examples/collocation.md",
            "examples/neural_gde.md",
            "examples/normalizing_flows.md"],
        "Bayesian Estimation Tutorials" => Any[
            "examples/turing_bayesian.md",
            "examples/BayesianNODE_NUTS.md",
            "examples/BayesianNODE_SGLD.md",
        ],
        "FAQ, Tips, and Tricks" => Any[
            "examples/local_minima.md",
            "examples/second_order_neural.md",
            "examples/second_order_adjoints.md",
            "examples/minibatch.md",
        ],
        "Hybrid and Jump Tutorials" => Any[
            "examples/hybrid_diffeq.md",
            "examples/jump.md",
        ],
        "Optimal and Model Predictive Control Tutorials" => Any[
            "examples/optimal_control.md",
            "examples/feedback_control.md",
        ],
        "Universal Differential Equations and Physical Constraints Tutorials" => Any[
            "examples/universal_diffeq.md",
            "examples/exogenous_input.md",
            "examples/physical_constraints.md",
            "examples/tensor_layer.md",
            "examples/hamiltonian_nn.md"
        ],
        "Layers" => Any[
        "Classical Basis Layers" => "layers/BasisLayers.md",
        "Tensor Product Layer" => "layers/TensorLayer.md",
        "Continuous Normalizing Flows Layer" => "layers/CNFLayer.md",
        "Spline Layer" => "layers/SplineLayer.md",
        "Neural Differential Equation Layers" => "layers/NeuralDELayers.md",
        "Hamiltonian Neural Network Layer" => "layers/HamiltonianNN.md"
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
