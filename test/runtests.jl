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
#         @safetestset "Neural Hamiltonian ODE Tests" begin
#             include("hamiltonian_nn.jl")
#         end
#     end

#     if GROUP == "All" || GROUP == "Aqua"
#         @safetestset "Aqua Q/A" begin
#             using Aqua, DiffEqFlux, LinearAlgebra
#             Aqua.find_persistent_tasks_deps(DiffEqFlux)
#             Aqua.test_ambiguities(DiffEqFlux; recursive = false)
#             #Aqua.test_deps_compat(DiffEqFlux)
#             Aqua.test_piracies(DiffEqFlux; treat_as_own = [LinearAlgebra.Tridiagonal])
#             Aqua.test_project_extras(DiffEqFlux)
#             Aqua.test_stale_deps(DiffEqFlux)
#             Aqua.test_unbound_args(DiffEqFlux)
#             Aqua.test_undefined_exports(DiffEqFlux)
#             # FIXME: Remove Tridiagonal piracy after
#             # https://github.com/JuliaDiff/ChainRules.jl/issues/713 is merged!
#         end
#     end
# end
