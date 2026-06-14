using SafeTestsets, Test
using SciMLTesting

run_tests(;
    core = () -> nothing,
    groups = Dict(
        "BasicNeuralDE" => function ()
            @safetestset "Neural DE" include("BasicNeuralDE/neural_de_tests.jl")
            @safetestset "Neural DAE" include("BasicNeuralDE/neural_dae_tests.jl")
            @safetestset "Neural ODE MM" include("BasicNeuralDE/neural_ode_mm_tests.jl")
            return @safetestset "Multiple Shooting" include("BasicNeuralDE/multiple_shoot_tests.jl")
        end,
        "AdvancedNeuralDE" => function ()
            @safetestset "CNF" include("AdvancedNeuralDE/cnf_tests.jl")
            return @safetestset "Second Order ODE" include("AdvancedNeuralDE/second_order_ode_tests.jl")
        end,
        "Newton" => function ()
            return @safetestset "Newton Neural ODE" include("Newton/newton_neural_ode_tests.jl")
        end,
        "Layers" => function ()
            @safetestset "Collocation" include("Layers/collocation_tests.jl")
            return @safetestset "Stiff Nested AD" include("Layers/stiff_nested_ad_tests.jl")
        end,
        "CUDA" => (;
            env = joinpath(@__DIR__, "CUDA"),
            body = joinpath(@__DIR__, "CUDA", "cuda_tests.jl"),
        ),
    ),
    qa = function ()
        return @safetestset "QA" include("QA/qa_tests.jl")
    end,
    all = ["BasicNeuralDE", "AdvancedNeuralDE", "Newton", "Layers"],
)
