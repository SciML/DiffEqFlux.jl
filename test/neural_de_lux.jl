using DiffEqFlux, Lux, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

rng = Random.default_rng()

mp = Float32[0.1,0.1]
x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,1.0f0)
dudt = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
luxdudt = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,2))

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
pd, st = Lux.setup(rng, node)
pd = Lux.ComponentArray(pd)
grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

#test with low tolerance ode solver
node = NeuralODE(luxdudt, tspan, Tsit5(), abstol=1e-12, reltol=1e-12, save_everystep=false, save_start=false)
grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
pd, st = Lux.setup(rng, node)
@test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
# @test ! iszero(grads[1])
# @test ! iszero(grads[2])

@test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
# @test ! iszero(grads[1])
# @test ! iszero(grads[2])

# goodgrad = grads[2]
# p = pd

node = NeuralODE(luxdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=BacksolveAdjoint())
grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
# goodgrad2 = grads[2]
# @test goodgrad ≈ goodgrad2 # Make sure adjoint overloads are correct

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
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0)
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.1)
    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    @test ! iszero(grads[1])
    @test ! iszero(grads[2])

    grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    @test !iszero(grads[1])
    @test !iszero(grads[2])
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
    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    # @test ! iszero(grads[1])
    # @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    # @test_broken ! iszero(grads[1])
    # @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0,sensealg=TrackerAdjoint())
    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    # @test ! iszero(grads[1])
    # @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    # @test_broken ! iszero(grads[1])
    # @test_broken ! iszero(grads[2])

    node = NeuralODE(luxdudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())
    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),x,pd,st)
    # @test ! iszero(grads[1])
    # @test ! iszero(grads[2])

    @test_broken grads = Zygote.gradient((x,p,st)->sum(node(x,p,st)[1]),xs,pd,st)
    # @test_broken ! iszero(grads[1])
    # @test_broken ! iszero(grads[2])
end

@info "Test non-ODEs"

dudt2 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
luxdudt2 = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,2))
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
pd, st = Lux.setup(rng, sode)

grads = Zygote.gradient((x,p,st)->sum(sode(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

grads = Zygote.gradient((x,p,st)->sum(sode(x,p,st)[1]),xs,pd,st)
@test_broken grads isa Tuple
@test ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

dudt22 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,4),x->reshape(x,2,2))
luxdudt22 = Lux.Chain(Lux.Dense(2,50,tanh),Lux.Dense(50,4),x->reshape(x,2,2))
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
pd,st = Lux.setup(rng, sode)

grads = Zygote.gradient((x,p,st)->sum(sode(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

@test_broken grads = Zygote.gradient((x,p,st)->sum(sode(x,p,st)),xs,pd,st)
@test_broken ! iszero(grads[1])
@test ! iszero(grads[2])
@test ! iszero(grads[2][end])

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
NeuralCDDE(luxddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(luxddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)
pd, st = Lux.setup(rng, dode)
grads = Zygote.gradient((x,p,st)->sum(dode(x,p,st)[1]),x,pd,st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

@test_broken grads = Zygote.gradient((x,p,st)->sum(dode(x,p,st)[1]),x,pd,st) isa Tuple
@test_broken ! iszero(grads[1])
@test ! iszero(grads[2])