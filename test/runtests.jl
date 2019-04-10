using DiffEqFlux, Test, SafeTestsets

@safetestset "Utils Tests" begin include("utils.jl") end
@safetestset "Layers Tests" begin include("layers.jl") end
@safetestset "Neural DE Tests" begin include("neural_de.jl") end
@safetestset "Partial Neural Tests" begin include("partial_neural.jl") end
