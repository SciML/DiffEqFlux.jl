using DiffEqFlux, Test, SafeTestsets

@safetestset "Utils Tests" begin include("utils.jl") end
@safetestset "Layers Tests" begin include("layers.jl") end
@safetestset "Layers SDE" begin include("layers_sde.jl") end
@safetestset "Layers DDE" begin include("layers_dde.jl") end
@safetestset "odenet" begin include("odenet.jl") end
@safetestset "Neural DE Tests" begin include("neural_de.jl") end
@safetestset "Partial Neural Tests" begin include("partial_neural.jl") end
