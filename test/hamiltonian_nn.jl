using DiffEqFlux, OrdinaryDiffEq, ReverseDiff, Test

# Checks for Shapes and Non-Zero Gradients
u0 = rand(Float32, 6, 1)

hnn = HamiltonianNN(Flux.Chain(Flux.Dense(6, 12, relu), Flux.Dense(12, 1)))
p = hnn.p

@test size(hnn(u0)) == (6, 1)

@test !iszero(ReverseDiff.gradient(p -> sum(hnn(u0, p)), p))

hnn = HamiltonianNN(Flux.Chain(Flux.Dense(6, 12, relu), Flux.Dense(12, 1)))
p = hnn.p

@test size(hnn(u0)) == (6, 1)

@test !iszero(ReverseDiff.gradient(p -> sum(hnn(u0, p)), p))

# Test Convergence on a toy problem
t = range(0.0f0, 1.0f0, length = 64)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)

hnn = HamiltonianNN(Flux.Chain(Flux.Dense(2, 16, relu), Flux.Dense(16, 1)))
p = hnn.p

opt = ADAM(0.01)
loss(x, y, p) = sum((hnn(x, p) .- y) .^ 2)

initial_loss = loss(data, target, p)

epochs = 100
for epoch = 1:epochs
    gs = ReverseDiff.gradient(p -> loss(data, target, p), p)
    Flux.Optimise.update!(opt, p, gs)
end

final_loss = loss(data, target, p)

@test initial_loss > final_loss

# Test output and gradient of NeuralHamiltonianDE Layer
tspan = (0.0f0, 1.0f0)

model = NeuralHamiltonianDE(
    hnn,
    tspan,
    Tsit5(),
    save_everystep = false,
    save_start = true,
    saveat = range(tspan[1], tspan[2], length = 10),
)
sol = Array(model(data[:, 1]))
@test size(sol) == (2, 10)

ps = Flux.params(model)
gs = Flux.gradient(() -> sum(Array(model(data[:, 1]))), ps)

@test !iszero(gs[model.p])
