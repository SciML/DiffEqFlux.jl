# Convolutional Neural ODE MNIST Classifier on GPU

Training a Convolutional Neural Net Classifier for **MNIST** using a neural
ordinary differential equation **NN-ODE** on **GPUs** with **Minibatching**.

For a step-by-step tutorial see the tutorial on the MNIST Neural ODE Classification Tutorial
using Fully Connected Layers.

```julia
using DiffEqFlux, Statistics,
    ComponentArrays, CUDA, Zygote, MLDatasets, OrdinaryDiffEq, Printf, Test, LuxCUDA, Random
using Optimization, OptimizationOptimisers
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs, batchview
using OneHotArrays

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
    Conv((4, 4), 64 => 64, relu; stride = 2, pad = 1), GroupNorm(64, 64),
    Conv((4, 4), 64 => 64; stride = 2, pad = 1))

dudt = Chain(Conv((3, 3), 64 => 64, tanh; stride = 1, pad = 1),
    Conv((3, 3), 64 => 64, tanh; stride = 1, pad = 1))

fc = Chain(GroupNorm(64, 64), x -> relu.(x), MeanPool((6, 6)),
    x -> reshape(x, (64, :)), Dense(64, 10))

nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

function DiffEqArray_to_Array(x)
    xarr = gdev(x)
    return xarr[:, :, :, :, 1]
end

# Build our over-all model topology
m = Chain(down,                 # (28, 28, 1, BS) -> (6, 6, 64, BS)
    nn_ode,               # (6, 6, 64, BS) -> (6, 6, 64, BS, 1)
    DiffEqArray_to_Array, # (6, 6, 64, BS, 1) -> (6, 6, 64, BS)
    fc)                   # (6, 6, 64, BS) -> (10, BS)
ps, st = Lux.setup(Random.default_rng(), m)
ps = ComponentArray(ps) |> gdev
st = st |> gdev

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img = x_train[1][:, :, :, 1:1] |> gdev
lab = x_train[2][:, 1:1] |> gdev

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

opt_func = OptimizationFunction((ps, _, x, y) -> loss_function(ps, x, y),
    Optimization.AutoZygote())
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
@test accuracy(m, zip(x_train, y_train), res.u, st) > 0.8
```

## Expected Output

```txt
Iter:   1 || Train Accuracy: 8.453 || Test Accuracy: 8.883
Iter:  11 || Train Accuracy: 14.773 || Test Accuracy: 14.967
Iter:  21 || Train Accuracy: 24.383 || Test Accuracy: 24.433
Iter:  31 || Train Accuracy: 38.820 || Test Accuracy: 38.000
Iter:  41 || Train Accuracy: 30.852 || Test Accuracy: 31.350
Iter:  51 || Train Accuracy: 29.852 || Test Accuracy: 29.433
Iter:  61 || Train Accuracy: 45.195 || Test Accuracy: 45.217
Iter:  71 || Train Accuracy: 70.336 || Test Accuracy: 68.850
Iter:  81 || Train Accuracy: 76.250 || Test Accuracy: 75.783
Iter:  91 || Train Accuracy: 80.867 || Test Accuracy: 81.017
Iter: 101 || Train Accuracy: 86.398 || Test Accuracy: 85.317
Iter: 111 || Train Accuracy: 90.852 || Test Accuracy: 90.650
Iter: 121 || Train Accuracy: 93.477 || Test Accuracy: 92.550
Iter: 131 || Train Accuracy: 93.320 || Test Accuracy: 92.483
Iter: 141 || Train Accuracy: 94.273 || Test Accuracy: 93.567
Iter: 151 || Train Accuracy: 94.531 || Test Accuracy: 93.583
Iter: 161 || Train Accuracy: 94.992 || Test Accuracy: 94.067
Iter: 171 || Train Accuracy: 95.398 || Test Accuracy: 94.883
Iter: 181 || Train Accuracy: 96.945 || Test Accuracy: 95.633
Iter: 191 || Train Accuracy: 96.430 || Test Accuracy: 95.750
Iter: 201 || Train Accuracy: 96.859 || Test Accuracy: 95.983
Iter: 211 || Train Accuracy: 97.359 || Test Accuracy: 96.500
Iter: 221 || Train Accuracy: 96.586 || Test Accuracy: 96.133
Iter: 231 || Train Accuracy: 96.992 || Test Accuracy: 95.833
Iter: 241 || Train Accuracy: 97.148 || Test Accuracy: 95.950
Iter: 251 || Train Accuracy: 96.422 || Test Accuracy: 95.950
Iter: 261 || Train Accuracy: 96.094 || Test Accuracy: 95.633
Iter: 271 || Train Accuracy: 96.719 || Test Accuracy: 95.767
Iter: 281 || Train Accuracy: 96.719 || Test Accuracy: 96.000
Iter: 291 || Train Accuracy: 96.609 || Test Accuracy: 95.817
Iter: 301 || Train Accuracy: 96.656 || Test Accuracy: 96.033
Iter: 311 || Train Accuracy: 97.594 || Test Accuracy: 96.500
Iter: 321 || Train Accuracy: 97.633 || Test Accuracy: 97.083
Iter: 331 || Train Accuracy: 98.008 || Test Accuracy: 97.067
Iter: 341 || Train Accuracy: 98.070 || Test Accuracy: 97.150
Iter: 351 || Train Accuracy: 97.875 || Test Accuracy: 97.050
Iter: 361 || Train Accuracy: 96.922 || Test Accuracy: 96.500
Iter: 371 || Train Accuracy: 97.188 || Test Accuracy: 96.650
Iter: 381 || Train Accuracy: 97.820 || Test Accuracy: 96.783
Iter: 391 || Train Accuracy: 98.156 || Test Accuracy: 97.567
Iter: 401 || Train Accuracy: 98.250 || Test Accuracy: 97.367
Iter: 411 || Train Accuracy: 97.969 || Test Accuracy: 97.267
Iter: 421 || Train Accuracy: 96.555 || Test Accuracy: 95.667
```
