using Documenter, Example, DiffEqFlux

# push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "DiffEqFlux.jl",
    clean = true, doctest = false,
    modules = [DiffEqFlux],
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md"
    ]
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
