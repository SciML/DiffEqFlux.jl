using DiffEqFlux, Test, Aqua, ExplicitImports

@testset "Aqua Q/A" begin
    Aqua.test_all(DiffEqFlux; ambiguities = false)
    Aqua.test_ambiguities(DiffEqFlux; recursive = false)
end

@testset "Explicit Imports" begin
    @test check_no_implicit_imports(
        DiffEqFlux; skip = (ADTypes, Lux, Base, Boltz, Core)
    ) === nothing
    @test check_no_stale_explicit_imports(DiffEqFlux) === nothing
end
