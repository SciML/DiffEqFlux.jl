using OrdinaryDiffEq, StochasticDiffEq, Flux, DiffEqSensitivity, DiffEqFlux, Zygote, Test, CuArrays
CuArrays.allowscalar(false)

mp = Chain(Dense(2,2)) |> gpu
x = Float32[2.; 0.] |> gpu
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.])) |> gpu
tspan = (0.0f0,25.0f0)
dudt = Chain(Dense(2,50,tanh),Dense(50,2)) |> gpu

NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)(x)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)(x)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())(x)

NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)(xs)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1)(xs)
NeuralODE(dudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())(xs)

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
grads = Zygote.gradient(()->sum(Array(node(x))),Flux.params(x,node))
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

# Adjoint
@testset "adjoint mode" begin
    node = NeuralODE(dudt,tspan,Tsit5(),save_everystep=false,save_start=false)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=0.0:0.1:10.0)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])

    node = NeuralODE(dudt,tspan,Tsit5(),saveat=1f-1)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
    @test ! iszero(grads[xs])
    @test ! iszero(grads[node.p])
end

NeuralDSDE(dudt,mp,(0.0f0,2.0f0),SOSRI(),saveat=0.0:0.1:2.0)(x)
sode = NeuralDSDE(dudt,mp,(0.0f0,2.0f0),SOSRI(),saveat=Float32.(0.0:0.1:2.0),dt=1f-1)
@test_broken grads = Zygote.gradient(()->sum(sode(x)),Flux.params(x,sode)) isa Tuple
@test ! iszero(grads[x])
@test ! iszero(grads[sode.p])

grads = Zygote.gradient(()->sum(sode(xs)),Flux.params(xs,sode))
@test ! iszero(grads[xs])
@test ! iszero(grads[sode.p])
=#
