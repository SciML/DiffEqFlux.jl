using Documenter, Example, DiffEqFlux

makedocs(
    sitename = "DiffEqFlux.jl",
    authors="Chris Rackauckas et al.",
    clean = true,
    doctest = false,
    modules = [DiffEqFlux],

    format = Documenter.HTML(#analytics = "",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Simple examples" => Any[
            "examples/LV-ODE.md",
            "examples/LV-delay.md",
            "examples/LV-stochastic.md",
            "examples/LV-GPU.md",
            "examples/LV-NN-ODE.md",
            "examples/LV-NN-SDE.md",
            "examples/LV-Univ.md",
            "examples/LV-NN-Stiff.md",
            "examples/LV-Jump.md"
        ],
        "API" => "API.md",
        "Benchmark" => "Benchmark.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
