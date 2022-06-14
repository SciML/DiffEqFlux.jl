using Lux, DiffEqFlux, DelayDiffEq, OrdinaryDiffEq, StochasticDiffEq, Test, Random

rng = Random.default_rng()
x = Float32[2.; 0.]
xs = Float32.(hcat([0.; 0.], [1.; 0.], [2.; 0.]))
tspan = (0.0f0, 1.0f0)
dudt = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4))
dudt2 = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 4))
dudt22 = Lux.Chain(Lux.Dense(4, 50, tanh), Lux.Dense(50, 16), ActivationFunction(x -> reshape(x, 4, 4)))
ddudt = Lux.Chain(Lux.Dense(12, 50, tanh), Lux.Dense(50, 4))

# Augmented Neural ODE
anode = AugmentedNDELayer(
    NeuralODE(dudt, tspan, Tsit5(), save_everystep=false, save_start=false), 2
)
p1, st1 = Lux.setup(rng, dudt)
p1 = Lux.ComponentArray(p1)
anode(x,p1,st1)
grads = Zygote.gradient((x, p, st) -> sum(anode(x, p, st)[1]), x, p1, st1)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

# Augmented Neural DSDE
andsde = AugmentedNDELayer(
    NeuralDSDE(dudt, dudt2, (0.0f0, 0.1f0), SOSRI(), saveat=0.0:0.01:0.1), 2
)
p2, st2 = Lux.setup(rng, dudt2)
p2 = Lux.ComponentArray(p2)
p = [p1,p2]
andsde(x,p,st1,st2)

grads = Zygote.gradient((x,p,st1,st2) -> sum(andsde(x,p,st1,st2)[1]),x,p,st1,st2)
@test ! iszero(grads[1])
@test ! iszero(grads[2])

# Augmented Neural SDE
asode = AugmentedNDELayer(
    NeuralSDE(dudt, dudt22,(0.0f0, 0.1f0), 4, LambaEM(), saveat=0.0:0.01:0.1), 2
)
p22, st22 = Lux.setup(rng,dudt22)
p22 = Lux.ComponentArray(p22)
p = [p1,p22]
asode(x,p,st1,st22)

ograds = Zygote.gradient((x,p,st1,st22) -> sum(asode(x,p,st1,st22)[1]),x,p,st1,st22)
@test ! iszero(grads[1])
@test ! iszero(grads[1])

# Augmented Neural CDDE
adode = AugmentedNDELayer(
    NeuralCDDE(ddudt, (0.0f0, 2.0f0), (p, t) -> zeros(Float32, 4), (1f-1, 2f-1),
               MethodOfSteps(Tsit5()), saveat=0.0:0.1:2.0), 2
)
p, st = Lux.setup(rng, ddudt)
p = Lux.ComponentArray(p)
adode(x,p,st)

grads = Zygote.gradient((x,p,st) -> sum(adode(x,p,st)[1]), x, p, st)
@test ! iszero(grads[1])
@test ! iszero(grads[2])
