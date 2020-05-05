using Documenter, Example, DiffEqFlux

# push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "DiffEqFlux.jl",
    authors="Chris Rackauckas et al.",    
    clean = true, doctest = false,
    modules = [DiffEqFlux],
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Examples" => Any[
            "examples/LV-ODE.md",
            "examples/LV-delay.md",
            "examples/LV-stochastic.md",
            "examples/LV-GPU.md",
            "examples/LV-NN-ODE.md",
            "examples/LV-NN-SDE.md",
            "examples/LV-Univ.md",
            "examples/LV-NN-Stiff.md",
            "examples/Unsorted.md"
        ]
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)


