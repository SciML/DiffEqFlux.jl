using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqFlux,
      Zygote, Test, DiffEqSensitivity, Tracker

x = Float32[2.; 0.]
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))
p = Flux.params(x,dudt)
neural_ode(dudt,x,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,x,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,x,tspan,Tsit5(),saveat=0.1)

grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
grads.grads
@test ! iszero(grads[x])
@test ! iszero(grads[dudt[1].W])
grads = Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
grads.grads
@test ! iszero(grads[x])
@test ! iszero(grads[dudt[1].W])

@test_broken Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false,sensealg=BacksolveAdjoint())),p) isa Zygote.Grads

# Adjoint
@testset "adjoint mode" begin
    grads = Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,saveat=0.0:0.1:10.0)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,saveat=0.1)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
end

# RD
@testset "reverse mode" begin
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,x,tspan,p=p,Tsit5(),saveat=0.0:0.1:10.0)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,x,tspan,p=p,Tsit5(),saveat=0.1)),p)
    @test ! iszero(grads[x])
    @test ! iszero(grads[dudt[1].W])
end

mp = Float32[0.1,0.1]
neural_dmsde(dudt,x,mp,(0.0f0,2.0f0),SOSRI(),saveat=0.1)
grads = Zygote.gradient(()->sum(neural_dmsde(dudt,x,mp,(0.0f0,2.0f0),SOSRI(),p=p,saveat=0.0:0.1:2.0)),p)
@test ! iszero(grads[x])
@test ! iszero(grads[dudt[1].W])

# Batch
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2))
p = Flux.params(xs,dudt)

neural_ode(dudt,xs,tspan,Tsit5(),save_everystep=false,save_start=false)
neural_ode(dudt,xs,tspan,Tsit5(),saveat=0.1)
neural_ode_rd(dudt,xs,tspan,Tsit5(),saveat=0.1)

grads = Zygote.gradient(()->sum(neural_ode(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
@test ! iszero(grads[xs])
@test ! iszero(grads[dudt[1].W])
grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,x,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
@test ! iszero(grads[xs])
@test ! iszero(grads[dudt[1].W])

# Adjoint
@testset "adjoint mode batches" begin
    grads = Zygote.gradient(()->sum(neural_ode(dudt,xs,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode(dudt,xs,tspan,Tsit5(),p=p,saveat=0.0:0.1:10.0)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode(dudt,xs,tspan,Tsit5(),p=p,saveat=0.1)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
end

# RD
@testset "reverse mode batches" begin
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),p=p,save_everystep=false,save_start=false)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),p=p,saveat=0.0:0.1:10.0)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
    grads = Zygote.gradient(()->sum(neural_ode_rd(dudt,xs,tspan,Tsit5(),p=p,saveat=0.1)),p)
    @test ! iszero(grads[xs])
    @test ! iszero(grads[dudt[1].W])
end
