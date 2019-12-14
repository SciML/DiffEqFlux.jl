using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqFlux, Zygote, Test

x = Float32[2.; 0.]
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)

Zygote.gradient(x->sum(neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)),x)

# Adjoint
@testset "adjoint mode" begin
    @test ! iszero(Zygote.gradient(x->sum(neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)),x)[1])
    @test ! iszero(Zygote.gradient(x->sum(neural_ode(dudt,x,tspan,Tsit5(),saveat=0.0:0.1:10.0)),x)[1])
    @test ! iszero(Zygote.gradient(x->sum(neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)),x)[1])
end

# RD
@testset "reverse mode" begin
    @test ! iszero(Zygote.gradient(x->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)),x)[1])
    @test ! iszero(Zygote.gradient(x->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.0:0.1:10.0)),x)[1])
    @test ! iszero(Zygote.gradient(x->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)),x)[1])
end

mp = Float32[0.1,0.1]
neural_dmsde(dudt,x,mp,(0.0f0,2.0f0),SOSRI(),saveat=0.1)
@test ! iszero(Zygote.gradient(x->sum(neural_dmsde(dudt,x,mp,(0.0f0,2.0f0),SOSRI(),saveat=0.0:0.1:2.0)),x)[1])

# Batch
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))

neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.1)

# Adjoint
@testset "adjoint mode batches" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    @test_broken Flux.back!(sum(neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.1)))
    #@test ! iszero(Tracker.grad(dudt[1].W))
end;

# RD
@testset "reverse mode batches" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Flux.back!(sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.1)))
    @test ! iszero(Tracker.grad(dudt[1].W))
end;


# Grads w.r.t. u0
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))
downsample = Dense(3,2)
x0 = Float32.(hcat([0.; 0; 0.], [1.,1.,1.],[2.,2.,2.]))
u0 = downsample(x0)

neural_ode(dudt,u0,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,u0,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,u0,tspan,Tsit5(),saveat=0.1)

# Adjoint
@testset "adjoint mode trackedu0" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    Tracker.zero_grad!(downsample.W.grad)
    m1 = Chain(downsample, u0->neural_ode(dudt,u0,tspan,Tsit5(),save_everystep=false,save_start=false))
    Flux.back!(sum(m1(x0)))
    @test ! iszero(Tracker.grad(dudt[1].W))
    @test ! iszero(Tracker.grad(downsample.W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Tracker.zero_grad!(downsample.W.grad)
    m2 = Chain(downsample, u0->neural_ode(dudt,u0,tspan,Tsit5(),saveat=0.0:0.1:10.0))
    Flux.back!(sum(m2(x0)))
    @test ! iszero(Tracker.grad(dudt[1].W))
    @test ! iszero(Tracker.grad(downsample.W))

    Tracker.zero_grad!(dudt[1].W.grad)
    Tracker.zero_grad!(downsample.W.grad)
    m3 = Chain(downsample, u0->neural_ode(dudt,u0,tspan,Tsit5(),saveat=0.1))
    @test_broken Flux.back!(sum(m3(x0)))
    #@test ! iszero(Tracker.grad(dudt[1].W))
    #@test ! iszero(Tracker.grad(downsample.W))
end;

# RD
@testset "reverse mode trackedu0" begin
    Tracker.zero_grad!(dudt[1].W.grad)
    u0 = downsample(x0)
    Flux.back!(sum(neural_ode_rd(dudt,u0,tspan,Tsit5(),save_everystep=false,save_start=false)))
    @test ! iszero(Tracker.grad(dudt[1].W))
    @test ! iszero(Tracker.grad(u0))

    Tracker.zero_grad!(dudt[1].W.grad)
    u0 = downsample(x0)
    Flux.back!(sum(neural_ode_rd(dudt,u0,tspan,Tsit5(),saveat=0.0:0.1:10.0)))
    @test ! iszero(Tracker.grad(dudt[1].W))
    @test ! iszero(Tracker.grad(u0))

    Tracker.zero_grad!(dudt[1].W.grad)
    u0 = downsample(x0)
    @test_broken Flux.back!(sum(neural_ode_rd(dudt,u0,tspan,Tsit5(),saveat=0.1)))
    #@test ! iszero(Tracker.grad(dudt[1].W))
    #@test ! iszero(Tracker.grad(downsample.W))
end;
