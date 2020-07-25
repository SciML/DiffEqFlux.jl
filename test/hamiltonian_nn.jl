using DiffEqFlux, Flux, OrdinaryDiffEq, ReverseDiff

# Checks for Shapes and Non-Zero Gradients
u0 = rand(Float32, 6, 1)

hnn = HamiltonianNN(Chain(Dense(6, 12, relu), Dense(12, 1)))

@test size(hnn(u0)) == (6, 1)

@test ! iszero(ReverseDiff.gradient(p -> sum(hnn(u0, p)), p))

hnn = HamiltonianNN(Chain(Dense(6, 12, relu), Dense(12, 1)))

@test size(hnn(u0)) == (6, 1)

@test ! iszero(ReverseDiff.gradient(p -> sum(hnn(u0, p)), p))

# Test Convergence on a toy problem
t = range(0.0f0, 1.0f0, length = 64)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)

hnn = HamiltonianNN(Chain(Dense(2, 16, relu), Dense(16, 1)))
p = hnn.p

opt = ADAM(0.01)
loss(x, y, p) = sum((hnn(x, p) .- y) .^ 2)

initial_loss = loss(data, target, p)

epochs = 100
for epoch in 1:epochs
    gs = ReverseDiff.gradient(p -> loss(data, target, p), p)
    Flux.Optimise.update!(opt, p, gs)
end

final_loss = loss(data, target, p)

@test initial_loss > final_loss