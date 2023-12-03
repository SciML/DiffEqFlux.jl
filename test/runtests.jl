using DiffEqFlux, SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = (Sys.iswindows() && haskey(ENV, "APPVEYOR"))
const is_CI = haskey(ENV, "CI")

@time begin
    if GROUP == "All" || GROUP == "DiffEqFlux" || GROUP == "Layers"
        @safetestset "Collocation" begin
            include("collocation.jl")
        end
        @safetestset "Stiff Nested AD Tests" begin
            include("stiff_nested_ad.jl")
        end
    end

    if GROUP == "All" || GROUP == "DiffEqFlux" || GROUP == "BasicNeuralDE"
        @safetestset "Neural DE Tests" begin
            include("neural_de.jl")
        end
        @safetestset "Neural Graph DE" begin
            include("neural_gde.jl")
        end
        @safetestset "Tensor Product Layer" begin
            include("tensor_product_test.jl")
        end
        @safetestset "Spline Layer" begin
            include("spline_layer_test.jl")
        end
        @safetestset "Multiple shooting" begin
            include("multiple_shoot.jl")
        end
        @safetestset "Neural ODE MM Tests" begin
            include("neural_ode_mm.jl")
        end
        # DAE Tests were never included
        # @safetestset "Neural DAE Tests" begin
        #     include("neural_dae.jl")
        # end
    end

    if GROUP == "All" || GROUP == "AdvancedNeuralDE"
        @safetestset "CNF Layer Tests" begin
            include("cnf_test.jl")
        end
        @safetestset "Neural Second Order ODE Tests" begin
            include("second_order_ode.jl")
        end
        @safetestset "Neural Hamiltonian ODE Tests" begin
            include("hamiltonian_nn.jl")
        end
    end

    if GROUP == "All" || GROUP == "Newton"
        @safetestset "Newton Neural ODE Tests" begin
            include("newton_neural_ode.jl")
        end
    end

    if !is_APPVEYOR && GROUP == "GPU"
        @safetestset "Neural DE GPU Tests" begin
            include("neural_de_gpu.jl")
        end
        @safetestset "MNIST GPU Tests: Fully Connected NN" begin
            include("mnist_gpu.jl")
        end
        @safetestset "MNIST GPU Tests: Convolutional NN" begin
            include("mnist_conv_gpu.jl")
        end
    end

    if GROUP == "All" || GROUP == "Aqua"
        @safetestset "Aqua Q/A" begin
            using Aqua, DiffEqFlux, LinearAlgebra
            Aqua.find_persistent_tasks_deps(DiffEqFlux)
            Aqua.test_ambiguities(DiffEqFlux, recursive = false)
            Aqua.test_deps_compat(DiffEqFlux)
            Aqua.test_piracies(DiffEqFlux; treat_as_own = [LinearAlgebra.Tridiagonal])
            Aqua.test_project_extras(DiffEqFlux)
            Aqua.test_stale_deps(DiffEqFlux)
            Aqua.test_unbound_args(DiffEqFlux)
            Aqua.test_undefined_exports(DiffEqFlux)
            # FIXME: Remove Tridiagonal piracy after
            # https://github.com/JuliaDiff/ChainRules.jl/issues/713 is merged!
        end
    end
end
