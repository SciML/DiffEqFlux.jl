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
        "Ordinary Differential Equation (ODE) Tutorials" => Any[
            "examples/optimization_ode.md",
            "examples/stiff_ode_fit.md",
            "examples/neural_ode_sciml.md",
            "examples/mnist_neural_ode.md",
            "examples/mnist_conv_neural_ode.md",
            "examples/augmented_neural_ode.md",
            "examples/collocation.md",
            "examples/neural_gde.md",
            "examples/exogenous_input.md",
            "examples/normalizing_flows.md"
        ],
        "Direct Usage with Optimizer Backends" => Any[
            "examples/neural_ode_galacticoptim.md",
            "examples/neural_ode_flux.md",
        ],
        "Training Techniques" => Any[
            "examples/multiple_shooting.md",
            "examples/local_minima.md",
            "examples/divergence.md",
            "examples/multiple_nn.md",
            "examples/data_parallel.md",
            "examples/second_order_neural.md",
            "examples/second_order_adjoints.md",
            "examples/minibatch.md",
        ],
        "Stochastic Differential Equation (SDE) Tutorials" => Any[
            "examples/optimization_sde.md",
            "examples/neural_sde.md",
        ],
        "Delay Differential Equation (DDE) Tutorials" => Any[
            "examples/delay_diffeq.md",
        ],
        "Differential-Algebraic Equation (DAE) Tutorials" => Any[
            "examples/physical_constraints.md",
        ],
        "Partial Differential Equation (PDE) Tutorials" => Any[
            "examples/pde_constrained.md",
        ],
        "Hybrid and Jump Equation Tutorials" => Any[
            "examples/hybrid_diffeq.md",
            "examples/bouncing_ball.md",
            "examples/jump.md",
        ],
        "Bayesian Estimation Tutorials" => Any[
            "examples/turing_bayesian.md",
            "examples/BayesianNODE_NUTS.md",
            "examples/BayesianNODE_SGLD.md",
        ],
        "Optimal and Model Predictive Control Tutorials" => Any[
            "examples/optimal_control.md",
            "examples/feedback_control.md",
            "examples/SDE_control.md",
        ],
        "Universal Differential Equations and Physical Layer Tutorials" => Any[
            "examples/universal_diffeq.md",
            "examples/tensor_layer.md",
            "examples/hamiltonian_nn.md"
        ],
        "Layer APIs" => Any[
        "Classical Basis Layers" => "layers/BasisLayers.md",
        "Tensor Product Layer" => "layers/TensorLayer.md",
        "Continuous Normalizing Flows Layer" => "layers/CNFLayer.md",
        "Spline Layer" => "layers/SplineLayer.md",
        "Neural Differential Equation Layers" => "layers/NeuralDELayers.md",
        "Hamiltonian Neural Network Layer" => "layers/HamiltonianNN.md"
        ],
        "Manual and APIs" => Any[
            "Controlling Choices of Adjoints" => "ControllingAdjoints.md",
            "Use with Flux Chain and train!" => "Flux.md",
            "FastChain" => "FastChain.md",
            "Smoothed Collocation" => "Collocation.md",
            "GPUs" => "GPUs.md",
            "sciml_train and GalacticOptim.jl" => "sciml_train.md",

        ],
        "Benchmarks" => "Benchmark.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
