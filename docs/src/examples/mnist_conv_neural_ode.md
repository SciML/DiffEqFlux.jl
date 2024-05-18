# Convolutional Neural ODE MNIST Classifier on GPU

Training a Convolutional Neural Net Classifier for **MNIST** using a neural
ordinary differential equation **NeuralODE** on **GPUs** with **Minibatching**.

For a step-by-step tutorial see the tutorial on the MNIST Neural ODE Classification Tutorial
using Fully Connected Layers.

```@example mnist_cnn
using DiffEqFlux, Statistics, ComponentArrays, CUDA, Zygote, MLDatasets, OrdinaryDiffEq,
      Printf, Test, LuxCUDA, Random
using Optimization, OptimizationOptimisers
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs, batchview

const cdev = cpu_device()
const gdev = gpu_device()

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

function loadmnist(batchsize = bs)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    function onehot(labels_raw)
        convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    end
    # Load MNIST
    mnist = MNIST(; split = :train)
    imgs, labels_raw = mnist.features, mnist.targets
    # Process images into (H,W,C,BS) batches
    x_train = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))) |>
              gdev
    x_train = batchview(x_train, batchsize)
    # Onehot and batch the labels
    y_train = onehot(labels_raw) |> gdev
    y_train = batchview(y_train, batchsize)
    return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Chain(Conv((3, 3), 1 => 64, relu; stride = 1), GroupNorm(64, 64),
    Conv((4, 4), 64 => 64, relu; stride = 2, pad = 1),
    GroupNorm(64, 64), Conv((4, 4), 64 => 64; stride = 2, pad = 1))

dudt = Chain(Conv((3, 3), 64 => 64, tanh; stride = 1, pad = 1),
    Conv((3, 3), 64 => 64, tanh; stride = 1, pad = 1))

fc = Chain(GroupNorm(64, 64), x -> relu.(x), MeanPool((6, 6)),
    x -> reshape(x, (64, :)), Dense(64, 10))

nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

function DiffEqArray_to_Array(x)
    xarr = gdev(x.u[1])
end

# Build our over-all model topology
m = Chain(down,                 # (28, 28, 1, BS) -> (6, 6, 64, BS)
    nn_ode,               # (6, 6, 64, BS) -> (6, 6, 64, BS, 1)
    DiffEqArray_to_Array, # (6, 6, 64, BS, 1) -> (6, 6, 64, BS)
    fc)                   # (6, 6, 64, BS) -> (10, BS)
ps, st = Lux.setup(Xoshiro(0), m)
ps = ComponentArray(ps) |> gdev
st = st |> gdev

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img = x_train[1][:, :, :, 1:1] |> gdev
lab = y_train[1][:, 1:1] |> gdev

x_m, _ = m(img, ps, st)

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 10)
    total_correct = 0
    total = 0
    st = Lux.testmode(st)
    for (x, y) in collect(data)[1:n_batches]
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(m, zip(x_train, y_train), ps, st)

function loss_function(ps, x, y)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

#burn in loss
loss_function(ps, x_train[1], y_train[1])

opt = OptimizationOptimisers.Adam(0.05)
iter = 0

opt_func = OptimizationFunction(
    (ps, _, x, y) -> loss_function(ps, x, y), Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps)

function callback(ps, l, pred)
    global iter += 1
    #Monitor that the weights do infact update
    #Every 10 training iterations show accuracy
    if (iter % 10 == 0)
        @info "[MNIST Conv GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps, st))"
    end
    return false
end

# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); maxiters = 10, callback)
acc = accuracy(m, zip(x_train, y_train), res.u, st)
@test acc > 0.8 # hide
acc # hide
```
