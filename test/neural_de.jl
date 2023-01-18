using DiffEqFlux, Zygote, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

mp = Float32[0.1,0.1]
x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,1.0f0)
dudt = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))

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
end

@info "Test non-ODEs"

dudt2 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
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

dudt22 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,4),x->reshape(x,2,2))
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

ddudt = Flux.Chain(Flux.Dense(6,50,tanh),Flux.Dense(50,2))
NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)

grads = Zygote.gradient(()->sum(dode(x)),Flux.params(x,dode))
@test ! iszero(grads[x])
@test ! iszero(grads[dode.p])

@test_broken grads = Zygote.gradient(()->sum(dode(xs)),Flux.params(xs,dode)) isa Tuple
@test_broken ! iszero(grads[xs])
@test ! iszero(grads[dode.p])

@testset "DimMover" begin
    r = rand(2, 3, 4, 5)
    @test r[:, :, 1, :] == FluxBatchOrder(r)[:, :, :, 1]
end
