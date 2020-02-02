using DiffEqFlux, Test, SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin
if GROUP == "All"
    @safetestset "Layers Tests" begin include("layers.jl") end
    @safetestset "Layers SciML Tests" begin include("layers_sciml.jl") end
    @safetestset "Layers SDE" begin include("layers_sde.jl") end
    @safetestset "Layers DDE" begin include("layers_dde.jl") end
    @safetestset "Size Handling in Adjoint Tests" begin include("size_handling_adjoint.jl") end
    @safetestset "odenet" begin include("odenet.jl") end
    @safetestset "GDP Regression Tests" begin include("gdp_regression_test.jl") end
    @safetestset "Neural DE Tests" begin include("neural_de.jl") end
    @safetestset "Partial Neural Tests" begin include("partial_neural.jl") end
end
end

if !is_APPVEYOR && GROUP == "GPU"
  @safetestset "odenet GPU" begin include("odenet_gpu.jl") end
  @safetestset "Neural DE GPU Tests" begin include("neural_de_gpu.jl") end
end
