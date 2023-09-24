using Documenter, DiffEqFlux

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["GKSwstype"] = "100"
using Plots
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

include("pages.jl")

makedocs(
    sitename = "DiffEqFlux.jl",
    authors="Chris Rackauckas et al.",
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    modules = [DiffEqFlux],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
                             canonical="https://docs.sciml.ai/DiffEqFlux/stable/"),
    pages=pages
)

deploydocs(
   repo = "github.com/SciML/DiffEqFlux.jl.git";
   push_preview = true
)
