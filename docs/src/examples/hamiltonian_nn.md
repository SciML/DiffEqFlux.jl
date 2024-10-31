# Hamiltonian Neural Network

Hamiltonian Neural Networks introduced in [1] allow models to "learn and respect exact conservation laws in an unsupervised manner". In this example, we will train a model to learn the Hamiltonian for a 1D Spring mass system. This system is described by the equation:

```math
m\ddot x + kx = 0
```

Now we make some simplifying assumptions, and assign ``m = 1`` and ``k = 1``. Analytically solving this equation, we get ``x = sin(t)``. Hence, ``q = sin(t)``, and ``p = cos(t)``. Using these solutions, we generate our dataset and fit the `NeuralHamiltonianDE` to learn the dynamics of this system.

## Copy-Pasteable Code

Before getting to the explanation, here's some code to start with. We will follow a full explanation of the definition and training process:

```@example hamiltonian_cp
using Lux, DiffEqFlux, OrdinaryDiffEq, Statistics, Plots, Zygote, ForwardDiff, Random,
      ComponentArrays, Optimization, OptimizationOptimisers, MLUtils

t = range(0.0f0, 1.0f0; length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t; dims = 1)
target = cat(dqdt, dpdt; dims = 1)
B = 256
NEPOCHS = 1000
dataloader = DataLoader((data, target); batchsize = B)

hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (64, 1)); autodiff = AutoZygote())
ps, st = Lux.setup(Xoshiro(0), hnn)
ps_c = ps |> ComponentArray

opt = OptimizationOptimisers.Adam(0.01f0)

function loss_function(ps, databatch)
    data, target = databatch
    pred, st_ = hnn(data, ps, st)
    return mean(abs2, pred .- target)
end

function callback(state, loss)
    println("[Hamiltonian NN] Loss: ", loss)
    return false
end

opt_func = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
opt_prob = OptimizationProblem(opt_func, ps_c, dataloader)

res = Optimization.solve(opt_prob, opt; callback, epochs = NEPOCHS)

ps_trained = res.u

model = NeuralODE(
    hnn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, save_start = true, saveat = t)

pred = Array(first(model(data[:, 1], ps_trained, st)))
plot(data[1, :], data[2, :]; lw = 4, label = "Original")
plot!(pred[1, :], pred[2, :]; lw = 4, label = "Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")
```

## Step by Step Explanation

### Data Generation

The HNN predicts the gradients ``(\dot q, \dot p)`` given ``(q, p)``. Hence, we generate the pairs ``(q, p)`` using the equations given at the top. Additionally, to supervise the training, we also generate the gradients. Next, we use Flux DataLoader for automatically batching our dataset.

```@example hamiltonian
using Lux, DiffEqFlux, OrdinaryDiffEq, Statistics, Plots, Zygote, ForwardDiff, Random,
      ComponentArrays, Optimization, OptimizationOptimisers, MLUtils

t = range(0.0f0, 1.0f0; length = 1024)
π_32 = Float32(π)
q_t = reshape(sin.(2π_32 * t), 1, :)
p_t = reshape(cos.(2π_32 * t), 1, :)
dqdt = 2π_32 .* p_t
dpdt = -2π_32 .* q_t

data = cat(q_t, p_t; dims = 1)
target = cat(dqdt, dpdt; dims = 1)
B = 256
NEPOCHS = 1000
dataloader = DataLoader((data, target); batchsize = B)
```

### Training the HamiltonianNN

We parameterize the  with a small MultiLayered Perceptron. HNNs are trained by optimizing the gradients of the Neural Network. Zygote currently doesn't support nesting itself, so we will be using ForwardDiff in the training loop to compute the gradients of the HNN Layer for Optimization.

```@example hamiltonian
hnn = Layers.HamiltonianNN{true}(Layers.MLP(2, (64, 1)); autodiff = AutoZygote())
ps, st = Lux.setup(Xoshiro(0), hnn)
ps_c = ps |> ComponentArray
hnn_stateful = StatefulLuxLayer{true}(hnn, ps_c, st)

opt = OptimizationOptimisers.Adam(0.01f0)

function loss_function(ps, databatch)
    (data, target) = databatch
    pred = hnn_stateful(data, ps)
    return mean(abs2, pred .- target), pred
end

function callback(state, loss)
    println("[Hamiltonian NN] Loss: ", loss)
    return false
end

opt_func = OptimizationFunction(loss_function, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_c, dataloader)

res = solve(opt_prob, opt; callback, epochs = NEPOCHS)

ps_trained = res.u
```

### Solving the ODE using trained HNN

In order to visualize the learned trajectories, we need to solve the ODE. We will use the
`NeuralODE` layer with `HamiltonianNN` layer, and solves the ODE.

```@example hamiltonian
model = NeuralODE(
    hnn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, save_start = true, saveat = t)

pred = Array(first(model(data[:, 1], ps_trained, st)))
plot(data[1, :], data[2, :]; lw = 4, label = "Original")
plot!(pred[1, :], pred[2, :]; lw = 4, label = "Predicted")
xlabel!("Position (q)")
ylabel!("Momentum (p)")
```

## References

[1] Greydanus, Samuel, Misko Dzamba, and Jason Yosinski. "Hamiltonian Neural Networks." Advances in Neural Information Processing Systems 32 (2019): 15379-15389.
