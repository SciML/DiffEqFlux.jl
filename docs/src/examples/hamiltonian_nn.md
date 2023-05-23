# Hamiltonian Neural Network

Hamiltonian Neural Networks introduced in [1] allow models to "learn and respect exact conservation laws in an unsupervised manner". In this example, we will train a model to learn the Hamiltonian for a 1D Spring mass system. This system is described by the equation:

```math
m\ddot x + kx = 0
```

Now we make some simplifying assumptions, and assign ``m = 1`` and ``k = 1``. Analytically solving this equation, we get ``x = sin(t)``. Hence, ``q = sin(t)``, and ``p = cos(t)``. Using these solutions, we generate our dataset and fit the `NeuralHamiltonianDE` to learn the dynamics of this system.

```@example hamiltonian_cp
using Lux, DiffEqFlux, OrdinaryDiffEq, Statistics, Plots, Zygote, ForwardDiff, Optimisers, Random, ComponentArrays

t = range(0.0f0, 1.0f0, length=1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = vcat(q_t, p_t)
target = vcat(dqdt, dpdt)
B = 256
dataloader = ((selectdim(data, 2, ((i-1)*B+1):(min(i * B, size(data, 2)))))
              for i in 1:(size(data, 2)÷B))

hnn = HamiltonianNN(Lux.Chain(Lux.Dense(2, 64, relu), Lux.Dense(64, 1)))
ps, st = Lux.setup(Random.default_rng(), hnn)
ps_c = ps |> ComponentArray

opt = ADAM(0.01f0)
st_opt = Optimisers.setup(opt, ps_c)

for epoch in 1:500
    for (x, y) in dataloader
        # Forward Mode over Reverse Mode for Training
        gs = ForwardDiff.gradient(ps_c -> mean(abs2, first(hnn(data, ps_c, st)) .- target), ps_c)
        st_opt, ps_c = Optimisers.update!(st_opt, ps_c, gs)
    end
    if epoch % 100 == 1
        println("Loss at epoch $epoch: $(mean(abs2, first(hnn(data, ps_c, st)) .- target))")
    end
end

model = NeuralHamiltonianDE(
    hnn, (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(first(model(data[:, 1], ps_c, st)))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")
```

## Step by Step Explanation

### Data Generation

The HNN predicts the gradients ``(\dot q, \dot p)`` given ``(q, p)``. Hence, we generate the pairs ``(q, p)`` using the equations given at the top. Additionally, to supervise the training, we also generate the gradients. Next, we use Flux DataLoader for automatically batching our dataset.

```@example hamiltonian
using Flux, DiffEqFlux, DifferentialEquations, Statistics, Plots, ReverseDiff

t = range(0.0f0, 1.0f0, length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t, dims = 1)
target = cat(dqdt, dpdt, dims = 1)
dataloader = ((selectdim(data, 2, ((i-1)*B+1):(min(i * B, size(data, 2)))))
              for i in 1:(size(data, 2)÷B))
```

### Training the HamiltonianNN

We parameterize the HamiltonianNN with a small MultiLayered Perceptron. HNNs are trained by optimizing the gradients of the Neural Network. Zygote currently doesn't support nesting itself, so we will be using ForwardDiff in the training loop to compute the gradients of the HNN Layer for Optimization.

```@example hamiltonian
hnn = HamiltonianNN(Lux.Chain(Lux.Dense(2, 64, relu), Lux.Dense(64, 1)))
ps, st = Lux.setup(Random.default_rng(), hnn)
ps_c = ps |> ComponentArray

opt = ADAM(0.01f0)
st_opt = Optimisers.setup(opt, ps_c)

for epoch in 1:500
    for (x, y) in dataloader
        # Forward Mode over Reverse Mode for Training
        gs = ForwardDiff.gradient(ps_c -> mean(abs2, first(hnn(data, ps_c, st)) .- target), ps_c)
        st_opt, ps_c = Optimisers.update!(st_opt, ps_c, gs)
    end
    if epoch % 100 == 1
        println("Loss at epoch $epoch: $(mean(abs2, first(hnn(data, ps_c, st)) .- target))")
    end
end
```

### Solving the ODE using trained HNN

In order to visualize the learned trajectories, we need to solve the ODE. We will use the `NeuralHamiltonianDE` layer, which is essentially a wrapper over `HamiltonianNN` layer, and solves the ODE.

```@example hamiltonian
model = NeuralHamiltonianDE(
    hnn, (0.0f0, 1.0f0),
    Tsit5(), save_everystep = false,
    save_start = true, saveat = t
)

pred = Array(first(model(data[:, 1], ps_c, st)))
plot(data[1, :], data[2, :], lw=4, label="Original")
plot!(pred[1, :], pred[2, :], lw=4, label="Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")
```

![HNN Prediction](https://user-images.githubusercontent.com/30564094/88309081-7cd76480-cd2b-11ea-981b-9cb86b153414.png)

## Expected Output

```julia
Loss at epoch 1: 16.079542
Loss at epoch 101: 0.017007854
Loss at epoch 201: 0.0118254945
Loss at epoch 301: 0.009933148
Loss at epoch 401: 0.009158727
```

## References

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks." Advances in Neural Information Processing Systems 32 (2019): 15379-15389.
