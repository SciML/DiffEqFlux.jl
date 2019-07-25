using OrdinaryDiffEq, Flux, DiffEqFlux, Test

@testset "NeuralODE struct" begin
    x = Float32[2.; 0.]
    tspan = (0.0f0,25.0f0)
    dudt = Chain(Dense(2,50,tanh),Dense(50,2))
    slver = Tsit5()
    slver_kwargs = Dict(:save_everystep=>false, :save_start=>false)

    funcsolv = neural_ode(dudt,x,tspan,slver;slver_kwargs...)
    
    node = NeuralODE(dudt,tspan,slver,slver_kwargs)

    @test node(x) ≈ funcsolv
end
