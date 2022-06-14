using DiffEqFlux, Lux, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

rng = Random.default_rng()

mp = Float32[0.1,0.1]
x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,1.0f0)
#Flux
dudt = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
#Lux
luxdudt = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,2))
p1, st1 = Lux.setup(rng, luxdudt)
p1 = Lux.ComponentArray(p1)

NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)(x)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)(x)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())(x)

NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)(xs)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)(xs)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())(xs)

@info "Test some gradients"

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=BacksolveAdjoint())
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()))
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

## Lux

@info "Test some Lux layers"

node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false)

grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

grads2 = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1)
@test ! iszero(grads2[1])
@test ! iszero(grads2[2])

#test with low tolerance ode solver
node = NeuralODE(luxdudt, tspan, Tsit5(), abstol=1e-12, reltol=1e-12, save_everystep=false, save_start=false)

grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
goodgrad = grads[2]

node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=BacksolveAdjoint())
grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)),x,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
goodgrad2 = grads[2]
@test goodgrad ≈ goodgrad2 # Make sure adjoint overloads are correct

@info "Test some adjoints"

# Adjoint
@testset "adjoint mode" begin
    node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.0:0.1:1.0)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false)
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0)
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.1)
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])
end

@info "Test Tracker"

# RD
@testset "Tracker mode" begin
    node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.0:0.1:1.0,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0,sensealg=TrackerAdjoint())
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)),x,p1,st1)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((xs,p,st)->sum(node(xs,p,st)[1]),xs,p1,st1) isa Tuple
    @test_broken ! iszero(grads[1])
    @test_broken ! iszero(grads[2])
end

@info "Test non-ODEs"
#Flux
dudt2 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
#Lux
luxdudt2 = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,2))
p2,st2 = Lux.setup(rng, luxdudt2)
p2 = Lux.ComponentArray(p2)
NeuralDSDE(dudt,dudt2,(0.0f0,.1f0),SOSRI(),saveat=0.1)(x)
sode = NeuralDSDE(dudt,dudt2,(0.0f0,.1f0),SOSRI(),saveat=0.0:0.01:0.1)

grads = Zygote.gradient(()->sum(sode(x)),Flux.params(x,sode))
@test ! iszero(grads[x])
@test ! iszero(grads[sode.p])
@test ! iszero(grads[sode.p][end])

grads = Zygote.gradient(()->sum(sode(xs)),Flux.params(xs,sode))
@test ! iszero(grads[xs])
@test ! iszero(grads[sode.p])
@test ! iszero(grads[sode.p][end])

sode = NeuralDSDE(luxdudt,luxdudt2,(0.0f0,.1f0),SOSRI(),saveat=0.0:0.01:0.1)
p = [p1,p2]
grads = Zygote.gradient((x,p,st1,st2)->sum(sode(x,p,st1,st2)[1]),x,p,st1,st2)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

grads2 = Zygote.gradient((xs,p,st1,st2)->sum(sode(xs,p,st1,st2)[1]),xs,p,st1,st2)
@test_broken grads2 isa Tuple
@test ! iszero(grads2[1])
@test ! iszero(grads2[2])
@test ! iszero(grads2[2][end])

dudt22 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,4),x->reshape(x,2,2))
luxdudt22 = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,4),ActivationFunction(x->reshape(x,2,2)))
p22,st22 = Lux.setup(rng, luxdudt22)
p22 = Lux.ComponentArray(p22)
NeuralSDE(dudt,dudt22,(0.0f0,.1f0),2,LambaEM(),saveat=0.01)(x)

sode = NeuralSDE(dudt,dudt22,(0.0f0,0.1f0),2,LambaEM(),saveat=0.0:0.01:0.1)

grads = Zygote.gradient(()->sum(sode(x)),Flux.params(x,sode))
@test ! iszero(grads[x])
@test ! iszero(grads[sode.p])
@test ! iszero(grads[sode.p][end])

@test_broken grads = Zygote.gradient(()->sum(sode(xs)),Flux.params(xs,sode))
@test_broken ! iszero(grads[xs])
@test ! iszero(grads[sode.p])
@test ! iszero(grads[sode.p][end])

sode = NeuralSDE(luxdudt,luxdudt22,(0.0f0,0.1f0),2,LambaEM(),saveat=0.0:0.01:0.1)
p = [p1,p22]
grads = Zygote.gradient((x,p,st1,st22)->sum(sode(x,p,st1,st22)[1]),x,p,st1,st22)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

@test_broken grads2 = Zygote.gradient((xs,p,st1,st22)->sum(sode(xs,p,st1,st22)[1]),xs,p,st1,st22)
@test_broken ! iszero(grads2[1])
@test ! iszero(grads2[2])
@test ! iszero(grads2[2][end])

ddudt = Flux.Chain(Flux.Dense(6,50,tanh),Flux.Dense(50,2))
NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)

grads = Zygote.gradient(()->sum(dode(x)),Flux.params(x,dode))
@test ! iszero(grads[x])
@test ! iszero(grads[dode.p])

@test_broken grads = Zygote.gradient(()->sum(dode(xs)),Flux.params(xs,dode)) isa Tuple
@test_broken ! iszero(grads[xs])
@test ! iszero(grads[dode.p])


luxddudt = Lux.Chain(Lux.Dense(6,50,tanh),Lux.Dense(50,2))
p,st = Lux.setup(rng, luxddudt)
p = Lux.ComponentArray(p)
NeuralCDDE(luxddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(luxddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)

grads = Zygote.gradient((x,p,st)->sum(dode(x,p,st)[1]),x,p,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

@test_broken grads2 = Zygote.gradient((xs,p,st)->sum(dode(xs,p,st)),xs,p,st) isa Tuple
@test_broken ! iszero(grads2[1])
@test ! iszero(grads2[2])