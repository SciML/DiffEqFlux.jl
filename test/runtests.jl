using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "QA"
        @time @safetestset "QA" include("qa_tests.jl")
    end
    if GROUP == "All" || GROUP == "BasicNeuralDE"
        @time @safetestset "Neural DE" include("neural_de_tests.jl")
        @time @safetestset "Neural DAE" include("neural_dae_tests.jl")
        @time @safetestset "Neural ODE MM" include("neural_ode_mm_tests.jl")
        @time @safetestset "Multiple Shooting" include("multiple_shoot_tests.jl")
    end
    if GROUP == "All" || GROUP == "AdvancedNeuralDE"
        @time @safetestset "CNF" include("cnf_tests.jl")
        @time @safetestset "Second Order ODE" include("second_order_ode_tests.jl")
    end
    if GROUP == "All" || GROUP == "Newton"
        @time @safetestset "Newton Neural ODE" include("newton_neural_ode_tests.jl")
    end
    if GROUP == "All" || GROUP == "Layers"
        @time @safetestset "Collocation" include("collocation_tests.jl")
        @time @safetestset "Stiff Nested AD" include("stiff_nested_ad_tests.jl")
    end
    if GROUP == "All" || GROUP == "CUDA"
        @time @safetestset "CUDA" include("cuda_tests.jl")
    end
end
