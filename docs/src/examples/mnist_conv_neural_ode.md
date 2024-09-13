# Convolutional Neural ODE MNIST Classifier on GPU

Training a Convolutional Neural Net Classifier for **MNIST** using a neural
ordinary differential equation **NeuralODE** on **GPUs** with **Minibatching**.

For a step-by-step tutorial see the tutorial on the MNIST Neural ODE Classification Tutorial
using Fully Connected Layers.

```@example mnist_cnn
using DiffEqFlux, ComponentArrays, CUDA, Zygote, MLDatasets, OrdinaryDiffEq,
      Printf, LuxCUDA, Random, MLUtils, OneHotArrays
using Optimization, OptimizationOptimisers
using MLDatasets: MNIST

const cdev = cpu_device()
const gdev = gpu_device()

logitcrossentropy = CrossEntropyLoss(; logits = Val(true))

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

function loadmnist(batchsize)
    # Load MNIST
    dataset = MNIST(; split = :train)[1:2000] # Partial load for demonstration
    imgs = dataset.features
    labels_raw = dataset.targets

    # Process images into (H, W, C, BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)

    return DataLoader(mapobs(gdev, (x_data, y_data)); batchsize, shuffle = true)
end

dataloader = loadmnist(128)

down = Chain(
    Conv((3, 3), 1 => 12, tanh; stride = 1),
    GroupNorm(12, 3),
    Conv((4, 4), 12 => 64, tanh; stride = 2, pad = 1),
    GroupNorm(64, 4),
    Conv((4, 4), 64 => 256; stride = 2, pad = 1)
)

dudt = Chain(
    Conv((3, 3), 256 => 64, tanh; pad = SamePad()),
    Conv((3, 3), 64 => 256, tanh; pad = SamePad())
)

fc = Chain(GroupNorm(256, 4, tanh), MeanPool((6, 6)), FlattenLayer(), Dense(256, 10))

nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
    reltol = 1e-5, abstol = 1e-6, save_start = false)

solution_to_array(sol) = sol.u[end]

m = Chain(
    down,
    nn_ode,
    solution_to_array,
    fc
)

ps, st = Lux.setup(Xoshiro(0), m);
ps = ComponentArray(ps) |> gdev;
st = st |> gdev;

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img, lab = first(dataloader);
x_m, _ = m(img, ps, st);

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 10)
    total_correct = 0
    total = 0
    st = Lux.testmode(st)
    for (x, y) in collect(data)[1:min(n_batches, length(data))]
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(m, ((img, lab),), ps, st)

function loss_function(ps, x, y)
    pred, _ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

# burn in loss
loss_function(ps, img, lab)

opt = OptimizationOptimisers.Adam(0.005)
iter = 0

opt_func = OptimizationFunction(
    (ps, _, x, y) -> loss_function(ps, x, y), Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps);

function callback(ps, l, pred)
    global iter += 1
    iter % 10 == 0 &&
        @info "[MNIST Conv GPU] Accuracy: $(accuracy(m, dataloader, ps.u, st))"
    return false
end

# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, dataloader; maxiters = 5, callback)
acc = accuracy(m, dataloader, res.u, st)
acc # hide
```
