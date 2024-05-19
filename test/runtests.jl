using ReTestItems

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All"
    ReTestItems.runtests(@__DIR__)
else
    tags = [Symbol(lowercase(GROUP))]
    ReTestItems.runtests(@__DIR__; tags)
end

#     if GROUP == "All" || GROUP == "AdvancedNeuralDE"
#         @safetestset "CNF Layer Tests" begin
#             include("cnf_test.jl")
#         end
#     end
