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
            "examples/LV-ODE.md",
            "examples/LV-stochastic.md",
            "examples/LV-Flux.md",
            "examples/LV-delay.md",
            "examples/NN-ODE.md",
            "examples/NeuralODE_Flux.md",
            "examples/Supervised-NN-ODE-MNIST.md",
            "examples/local_minima.md",
            "examples/NN-SDE.md",
            "examples/NeuralOptimalControl.md",
            "examples/LV-Univ.md",
            "examples/SecondOrderNeural.md",
            "examples/LV-NN-Stiff.md",
            "examples/NewtonSecondOrderAdjoints.md",
            "examples/LV-Jump.md",
            "examples/universaldiffeq.md",
            "examples/minibatch.md"
        ],
        "Continuous Normalizing Flows Layer" => "CNFLayer.md"
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
