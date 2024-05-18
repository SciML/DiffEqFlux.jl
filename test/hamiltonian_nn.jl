using DiffEqFlux, Zygote, OrdinaryDiffEq, ForwardDiff, Test, Optimisers, Random, Lux,
      ComponentArrays, Statistics

# Checks for Shapes and Non-Zero Gradients
u0 = rand(Float32, 6, 1)

for ad in (AutoForwardDiff(), AutoZygote())
    hnn = HamiltonianNN(Chain(Dense(6 => 12, relu), Dense(12 => 1)); ad)
    ps, st = Lux.setup(Xoshiro(0), hnn)
    ps = ps |> ComponentArray

    @test size(first(hnn(u0, ps, st))) == (6, 1)

    @test !iszero(ForwardDiff.gradient(ps -> sum(first(hnn(u0, ps, st))), ps))

    ad isa AutoZygote && continue

    @test !iszero(only(Zygote.gradient(ps -> sum(first(hnn(u0, ps, st))), ps)))
end

# Test Convergence on a toy problem
t = range(0.0f0, 1.0f0; length = 64)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = vcat(q_t, p_t)
target = vcat(dqdt, dpdt)

hnn = HamiltonianNN(Chain(Dense(2 => 16, relu), Dense(16 => 1)); ad = AutoForwardDiff())
ps, st = Lux.setup(Xoshiro(0), hnn)
ps = ps |> ComponentArray

opt = Optimisers.Adam(0.01)
st_opt = Optimisers.setup(opt, ps)
loss(data, target, ps) = mean(abs2, first(hnn(data, ps, st)) .- target)

initial_loss = loss(data, target, ps)

for epoch in 1:100
    global ps, st_opt
    gs = last(Zygote.gradient(loss, data, target, ps))
    st_opt, ps = Optimisers.update!(st_opt, ps, gs)
end

final_loss = loss(data, target, ps)

@test initial_loss > 5 * final_loss

# Test output and gradient of NeuralHamiltonianDE Layer
tspan = (0.0f0, 1.0f0)

model = NeuralHamiltonianDE(hnn, tspan, Tsit5(); save_everystep = false, save_start = true,
    saveat = range(tspan[1], tspan[2]; length = 10))
sol = Array(first(model(data[:, 1], ps, st)))
@test size(sol) == (2, 10)

gs = only(Zygote.gradient(ps -> sum(Array(first(model(data[:, 1], ps, st)))), ps))

@test !iszero(gs)
