using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqFlux, Test

x = Float32[2.; 0.]
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)

# Adjoint
@testset "adjoint mode" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode(dudt,x,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)))
    # @test ! iszero(Tracker.grad(dudt[1].W))
end;

# RD
@testset "reverse mode" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)))
    @test ! iszero(Tracker.grad(dudt[1].W))
end;

mp = Float32[0.1,0.1]
Tracker.zero_grad!(dudt[1].W.grad)
neural_dmsde(dudt,x,mp,tspan,SOSRI(),saveat=0.1)
Flux.back!(sum(neural_dmsde(dudt,x,mp,tspan,SOSRI(),saveat=0.1)))
@test ! iszero(Tracker.grad(dudt[1].W))


# Batch
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)

# Adjoint
@testset "adjoint mode batches" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test_broken ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test_broken ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.1)))
    @test_broken ! iszero(Tracker.grad(dudt[1].W))
end;

# RD
@testset "reverse mode batches" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.1)))
    @test ! iszero(Tracker.grad(dudt[1].W))
end;
