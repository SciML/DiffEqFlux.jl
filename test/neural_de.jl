using DiffEqFlux, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

mp = Float32[0.1,0.1]
x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0,1.0f0)
dudt = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
fastdudt = FastChain(FastDense(2,50,tanh),FastDense(50,2))

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

## Fast

@info "Test some fast layers"

node = NeuralODE(fastdudt,tspan,Tsit5(),save_everystep=false,save_start=false)
pd = Flux.params(node)[1]
gradsnc = Zygote.gradient(()->sum(node(x)),Flux.params(x,node)) # no cache
@test ! iszero(gradsnc[x])
@test ! iszero(gradsnc[node.p])

gradsnc2 = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(gradsnc2[xs])
@test ! iszero(gradsnc2[node.p])

gradsc2 = Zygote.gradient(()->sum(nodec(xs)),Flux.params(xs,nodec))
@test ! iszero(gradsc2[xs])
@test ! iszero(gradsc2[nodec.p])
@test gradsnc2[xs] ≈ gradsc2[xs] rtol=1e-6
@test gradsnc2[node.p] ≈ gradsc2[nodec.p] rtol=1e-6
#test with low tolerance ode solver
node = NeuralODE(fastdudt, tspan, Tsit5(), abstol=1e-12, reltol=1e-12, save_everystep=false, save_start=false)
pd = Flux.params(node)[1]
gradsnc = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(gradsnc[x])
@test ! iszero(gradsnc[node.p])

node = NeuralODE(fastdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

goodgrad = grads[node.p]
p = node.p

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])

goodgradc = grads[node.p]
pc = node.p

node = NeuralODE(fastdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=BacksolveAdjoint(),p=p)
grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
@test ! iszero(grads[x])
@test ! iszero(grads[node.p])

grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node))
@test ! iszero(grads[xs])
@test ! iszero(grads[node.p])
goodgrad2 = grads[node.p]
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

    node = NeuralODE(fastdudt,tspan,Tsit5(),save_everystep=false,save_start=false)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])

    node = NeuralODE(fastdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])

    node = NeuralODE(fastdudt,tspan,Tsit5(),saveat=0.1)
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])
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

    node = NeuralODE(fastdudt,tspan,Tsit5(),save_everystep=false,save_start=false,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])

    node = NeuralODE(fastdudt,tspan,Tsit5(),saveat=0.0:0.1:1.0,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])

    node = NeuralODE(fastdudt,tspan,Tsit5(),saveat=0.1,sensealg=TrackerAdjoint())
    grads = Zygote.gradient(()->sum(node(x)),Flux.params(x,node))
    @test ! iszero(grads[x])
    @test ! iszero(grads[node.p])

    @test_broken grads = Zygote.gradient(()->sum(node(xs)),Flux.params(xs,node)) isa Tuple
    @test_broken ! iszero(grads[xs])
    @test_broken ! iszero(grads[node.p])
end

@info "Test non-ODEs"

dudt2 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,2))
fastdudt2 = FastChain(FastDense(2,50,tanh),FastDense(50,2))
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

sode = NeuralDSDE(fastdudt,fastdudt2,(0.0f0,.1f0),SOSRI(),saveat=0.0:0.01:0.1)
pd = Flux.params(sode)[1]
Random.seed!(1234)
gradsnc = Zygote.gradient(()->sum(sode(x)),Flux.params(x,sode)) #no cacahe
@test ! iszero(gradsnc[x])
@test ! iszero(gradsnc[sode.p])
@test ! iszero(gradsnc[sode.p][end])

gradsnc2 = Zygote.gradient(()->sum(sode(xs)),Flux.params(xs,sode))
@test_broken gradsnc2 isa Tuple
@test ! iszero(gradsnc2[xs])
@test ! iszero(gradsnc2[sode.p])
@test ! iszero(gradsnc2[sode.p][end])

gradsc2 = Zygote.gradient(()->sum(sodec(xs)),Flux.params(xs,sodec))
@test_broken gradsc2 isa Tuple
@test ! iszero(gradsc2[xs])
@test ! iszero(gradsc2[sodec.p])
@test ! iszero(gradsc2[sodec.p][end])

dudt22 = Flux.Chain(Flux.Dense(2,50,tanh),Flux.Dense(50,4),x->reshape(x,2,2))
fastdudt22 = FastChain(FastDense(2,50,tanh),FastDense(50,4),(x,p)->reshape(x,2,2))
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

sode = NeuralSDE(fastdudt,fastdudt22,(0.0f0,0.1f0),2,LambaEM(),saveat=0.0:0.01:0.1)
pd = Flux.params(sode)[1]
Random.seed!(1234)
gradsnc = Zygote.gradient(()->sum(sode(x)),Flux.params(x,sode))
@test ! iszero(gradsnc[x])
@test ! iszero(gradsnc[sode.p])
@test ! iszero(gradsnc[sode.p][end])

@test_broken gradsnc = Zygote.gradient(()->sum(sode(xs)),Flux.params(xs,sode))
@test_broken ! iszero(gradsnc[xs])
@test ! iszero(gradsnc[sode.p])
@test ! iszero(gradsnc[sode.p][end])

@test_broken gradsc = Zygote.gradient(()->sum(sodec(xs)),Flux.params(xs,sodec))
@test_broken ! iszero(gradsc[xs])
@test ! iszero(gradsc[sodec.p])
@test ! iszero(gradsc[sodec.p][end])

ddudt = Flux.Chain(Flux.Dense(6,50,tanh),Flux.Dense(50,2))
NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(ddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)

grads = Zygote.gradient(()->sum(dode(x)),Flux.params(x,dode))
@test ! iszero(grads[x])
@test ! iszero(grads[dode.p])

@test_broken grads = Zygote.gradient(()->sum(dode(xs)),Flux.params(xs,dode)) isa Tuple
@test_broken ! iszero(grads[xs])
@test ! iszero(grads[dode.p])


fastddudt = FastChain(FastDense(6,50,tanh),FastDense(50,2))
NeuralCDDE(fastddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dode = NeuralCDDE(fastddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0)
pd = Flux.params(dode)[1]
gradsnc = Zygote.gradient(()->sum(dode(x)),Flux.params(x,dode))
@test ! iszero(gradsnc[x])
@test ! iszero(gradsnc[dode.p])

@test_broken gradsnc = Zygote.gradient(()->sum(dode(xs)),Flux.params(xs,dode)) isa Tuple
@test_broken ! iszero(gradsnc[xs])
@test ! iszero(gradsnc[dode.p])

fastcddudt = FastChain(FastDense(6,50,tanh,numcols=size(xs)[2],precache=true),FastDense(50,2,numcols=size(xs)[2],precache=true))
NeuralCDDE(fastcddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.1)(x)
dodec = NeuralCDDE(fastcddudt,(0.0f0,2.0f0),(p,t)->zero(x),(1f-1,2f-1),MethodOfSteps(Tsit5()),saveat=0.0:0.1:2.0,p=pd)

gradsc = Zygote.gradient(()->sum(dodec(x)),Flux.params(x,dodec))
@test ! iszero(gradsc[x])
@test ! iszero(gradsc[dodec.p])
@test gradsnc[x] ≈ gradsc[x] rtol=1e-6
@test gradsnc[dode.p] ≈ gradsc[dodec.p] rtol=1e-6

@test_broken gradsc = Zygote.gradient(()->sum(dodec(xs)),Flux.params(xs,dodec)) isa Tuple
@test_broken ! iszero(gradsc[xs])
@test ! iszero(gradsc[dodec.p])
