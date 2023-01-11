using DiffEqFlux, Lux, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0, 1.0f0)
fastdudt = FastChain(FastDense(4, 50, tanh), FastDense(50, 4))
fastdudt2 = FastChain(FastDense(4, 50, tanh), FastDense(50, 4))
fastdudt22 = FastChain(FastDense(4, 50, tanh), FastDense(50, 16), (x, p) -> reshape(x, 4, 4))
fastddudt = FastChain(FastDense(12, 50, tanh), FastDense(50, 4))

# Augmented Neural ODE
anode = AugmentedNDELayer(
    NeuralODE(fastdudt, tspan, Tsit5(), save_everystep=false, save_start=false), 2
)
anode(x)

grads = Zygote.gradient(() -> sum(anode(x)), Flux.params(x, anode.nde))
@test ! iszero(grads[x])
@test ! iszero(grads[anode.p])

# Augmented Neural DSDE
andsde = AugmentedNDELayer(
    NeuralDSDE(fastdudt, fastdudt2, (0.0f0, 0.1f0), SOSRI(), saveat=0.0:0.01:0.1), 2
)
andsde(x)

grads = Zygote.gradient(() -> sum(andsde(x)), Flux.params(x, andsde.nde))
@test ! iszero(grads[x])
@test ! iszero(grads[andsde.p])

# Augmented Neural SDE
asode = AugmentedNDELayer(
    NeuralSDE(fastdudt, fastdudt22,(0.0f0, 0.1f0), 4, LambaEM(), saveat=0.0:0.01:0.1), 2
)
asode(x)

grads = Zygote.gradient(() -> sum(asode(x)), Flux.params(x, asode.nde))
@test ! iszero(grads[x])
@test ! iszero(grads[asode.p])

# Augmented Neural CDDE
adode = AugmentedNDELayer(
    NeuralCDDE(fastddudt, (0.0f0, 2.0f0), (p, t) -> zeros(Float32, 4), (1f-1, 2f-1),
               MethodOfSteps(Tsit5()), saveat=0.0:0.1:2.0), 2
)
adode(x)

grads = Zygote.gradient(() -> sum(adode(x)), Flux.params(x, adode.nde))
@test ! iszero(grads[x])
@test ! iszero(grads[adode.p])

## AugmentedNDELayer with Lux

rng = Random.default_rng()

dudt = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4))
dudt2 = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4))
dudt22 = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 16), (x) -> reshape(x, 4, 4))

# Augmented Neural ODE
anode = AugmentedNDELayer(
    NeuralODE(dudt, tspan, Tsit5(), save_everystep=false, save_start=false), 2
)
pd, st = Lux.setup(rng, anode)
pd = Lux.ComponentArray(pd)
anode(x,pd,st)

grads = Zygote.gradient((x,p,st) -> sum(anode(x,p,st)[1]), x, pd, st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

# Augmented Neural DSDE
andsde = AugmentedNDELayer(
    NeuralDSDE(dudt, dudt2, (0.0f0, 0.1f0), EulerHeun(), saveat=0.0:0.01:0.1, dt=0.01), 2
)
pd, st = Lux.setup(rng, andsde)
pd = Lux.ComponentArray(pd)
andsde(x,pd,st)

grads = Zygote.gradient((x,p,st) -> sum(andsde(x,p,st)[1]), x, pd, st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

# Augmented Neural SDE
asode = AugmentedNDELayer(
    NeuralSDE(dudt, dudt22,(0.0f0, 0.1f0), 4, EulerHeun(), saveat=0.0:0.01:0.1, dt=0.01), 2
)
pd, st = Lux.setup(rng, asode)
pd = Lux.ComponentArray(pd)
asode(x,pd,st)

grads = Zygote.gradient((x,p,st) -> sum(asode(x,p,st)[1]), x, pd, st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
