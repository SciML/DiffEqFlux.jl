using DiffEqFlux
using Pkg
using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )
const is_CI = haskey(ENV,"CI")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
if GROUP == "All" || GROUP == "DiffEqFlux" || GROUP == "Layers"
    @safetestset "Layers Tests" begin include("layers.jl") end
    @safetestset "Fast Layers" begin include("fast_layers.jl") end
    @safetestset "Layers SciML Tests" begin include("layers_sciml.jl") end
    @safetestset "Layers SDE" begin include("layers_sde.jl") end
    @safetestset "Layers DDE" begin include("layers_dde.jl") end
    @safetestset "Collocation Regression" begin include("collocation_regression.jl") end
end

if GROUP == "All" || GROUP == "DiffEqFlux" || GROUP == "BasicNeuralDE"
    @safetestset "Neural DE Tests" begin include("neural_de.jl") end
    @safetestset "Augmented Neural DE Tests" begin include("augmented_nde.jl") end
    @safetestset "Neural Graph DE" begin include("neural_gde.jl") end
    @safetestset "Hybrid DE" begin include("hybrid_de.jl") end
    @safetestset "Neural ODE MM Tests" begin include("neural_ode_mm.jl") end
    @safetestset "Fast Neural ODE Tests" begin include("fast_neural_ode.jl") end
    @safetestset "Tensor Product Layer" begin include("tensor_product_test.jl") end
    @safetestset "Spline Layer" begin include("spline_layer_test.jl") end
end

if GROUP == "All" || GROUP == "AdvancedNeuralDE"
    @safetestset "CNF Layer Tests" begin include("cnf_test.jl") end
    @safetestset "Newton Neural ODE Tests" begin include("newton_neural_ode.jl") end
    @safetestset "Neural Second Order ODE Tests" begin include("second_order_ode.jl") end
    @safetestset "Neural Hamiltonian ODE Tests" begin include("hamiltonian_nn.jl") end
end

if GROUP == "All" || GROUP == "Integration"
    @safetestset "Ensemble Tests" begin include("ensembles.jl") end
    @safetestset "Partial Neural Tests" begin include("partial_neural.jl") end
    @safetestset "Size Handling in Adjoint Tests" begin include("size_handling_adjoint.jl") end
    @safetestset "odenet" begin include("odenet.jl") end
    @safetestset "GDP Regression Tests" begin include("gdp_regression_test.jl") end
    @safetestset "Stiff Nested AD Tests" begin include("stiff_nested_ad.jl") end
end
end

if !is_APPVEYOR && GROUP == "GPU"
    if is_CI
        activate_gpu_env()
    end
    @safetestset "odenet GPU" begin include("odenet_gpu.jl") end
    @safetestset "Neural DE GPU Tests" begin include("neural_de_gpu.jl") end
    @safetestset "MNIST GPU Tests" begin include("mnist_gpu.jl") end
end
