using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqFlux, Test

xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

@testset "only end" begin
    only_end = neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)
    @test_broken size(only_end)[end]==1
end


