@testitem "Aqua Q/A" tags=[:qa] begin
    using Aqua

    Aqua.test_all(DiffEqFlux; ambiguities = false)
    Aqua.test_ambiguities(DiffEqFlux; recursive = false)
end

@testitem "Explicit Imports" tags=[:qa] begin
    using ExplicitImports

    @test check_no_implicit_imports(
        DiffEqFlux; skip = (ADTypes, Lux, Base, Boltz, Core)) === nothing
    @test check_no_stale_explicit_imports(DiffEqFlux) === nothing
end
