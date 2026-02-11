using DiffEqFlux, Lux, ComponentArrays, Zygote, Optimization, OptimizationOptimisers,
    OrdinaryDiffEq, Test, Random
using DiffEqFlux: group_ranges

# SKIP: Test segfaults (signal 11) on Julia 1.11.
# See https://github.com/SciML/DiffEqFlux.jl/issues/1004
@info "Skipping Multiple Shooting tests due to segfault (issue #1004)"
@testset "Multiple Shooting" begin
    @test_broken false
end
