using DiffEqFlux,
    Lux, CUDA, Zygote, OrdinaryDiffEq, StochasticDiffEq, Test, Random,
    ComponentArrays

CUDA.allowscalar(false)

x = Float32[2.0; 0.0] |> Lux.gpu
xs = hcat([0.0; 0.0], [1.0; 0.0], [2.0; 0.0]) |> Lux.gpu
tspan = (0.0f0, 25.0f0)

mp = Lux.Chain(Lux.Dense(2, 2))

dudt = Lux.Chain(Lux.Dense(2, 50, tanh), Lux.Dense(50, 2))
ps_dudt, st_dudt = Lux.setup(Random.default_rng(), dudt)
ps_dudt = ComponentArray(ps_dudt) |> Lux.gpu
st_dudt = st_dudt |> Lux.gpu

NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false)(x,
    ps_dudt,
    st_dudt)
NeuralODE(dudt, tspan, Tsit5(), saveat = 0.1)(x, ps_dudt, st_dudt)
NeuralODE(dudt, tspan, Tsit5(), saveat = 0.1, sensealg = TrackerAdjoint())(x,
    ps_dudt,
    st_dudt)

NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false)(xs,
    ps_dudt,
    st_dudt)
NeuralODE(dudt, tspan, Tsit5(), saveat = 0.1)(xs, ps_dudt, st_dudt)
NeuralODE(dudt, tspan, Tsit5(), saveat = 0.1, sensealg = TrackerAdjoint())(xs,
    ps_dudt,
    st_dudt)

node = NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false)
ps_node, st_node = Lux.setup(Random.default_rng(), node)
ps_node = ComponentArray(ps_node) |> Lux.gpu
st_node = st_node |> Lux.gpu
grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

node = NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false,
    sensealg = TrackerAdjoint())
grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

node = NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false,
    sensealg = BacksolveAdjoint())
grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
@test !iszero(grads[1])
@test !iszero(grads[2])

# Adjoint
@testset "adjoint mode" begin
    node = NeuralODE(dudt, tspan, Tsit5(), save_everystep = false, save_start = false)
    grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    node = NeuralODE(dudt, tspan, Tsit5(), saveat = 0.0:0.1:10.0)
    grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    node = NeuralODE(dudt, tspan, Tsit5(), saveat = 1.0f-1)
    grads = Zygote.gradient((x, ps) -> sum(first(node(x, ps, st_node))), x, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])

    grads = Zygote.gradient((xs, ps) -> sum(first(node(xs, ps, st_node))), xs, ps_node)
    @test !iszero(grads[1])
    @test !iszero(grads[2])
end

ndsde = NeuralDSDE(dudt, mp, (0.0f0, 2.0f0), SOSRI(), saveat = 0.0:0.1:2.0)
ps_ndsde, st_ndsde = Lux.setup(Random.default_rng(), ndsde)
ps_ndsde = ComponentArray(ps_ndsde) |> Lux.gpu
st_ndsde = st_ndsde |> Lux.gpu
ndsde(x, ps_ndsde, st_ndsde)

sode = NeuralDSDE(dudt, mp, (0.0f0, 2.0f0), SOSRI(), saveat = Float32.(0.0:0.1:2.0),
    dt = 1.0f-1, sensealg = TrackerAdjoint())
ps_sode, st_sode = Lux.setup(Random.default_rng(), sode)
ps_sode = ComponentArray(ps_sode) |> Lux.gpu
st_sode = st_sode |> Lux.gpu
grads = Zygote.gradient((x, ps) -> sum(first(sode(x, ps, st_sode))), x, ps_sode)
@test !iszero(grads[1])
@test !iszero(grads[2])

grads = Zygote.gradient((xs, ps) -> sum(first(sode(xs, ps, st_sode))), xs, ps_sode)
@test !iszero(grads[1])
@test !iszero(grads[2])
