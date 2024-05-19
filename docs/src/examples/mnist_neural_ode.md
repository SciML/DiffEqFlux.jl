# [GPU-based MNIST Neural ODE Classifier](@id mnist)

Training a classifier for **MNIST** using a neural ordinary differential equation **NeuralODE**
on **GPUs** with **minibatching**.

(Step-by-step description below)

```@example mnist
using DiffEqFlux, CUDA, Zygote, MLDataUtils, NNlib, OrdinaryDiffEq, Test, Lux, Statistics,
      ComponentArrays, Random, Optimization, OptimizationOptimisers, LuxCUDA
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const cdev = cpu_device()
const gdev = gpu_device()

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

down = Lux.Chain(Lux.FlattenLayer(), Lux.Dense(784, 20, tanh))
nn = Lux.Chain(Lux.Dense(20, 10, tanh), Lux.Dense(10, 10, tanh), Lux.Dense(10, 20, tanh))
fc = Lux.Dense(20, 10)

nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

function DiffEqArray_to_Array(x)
    xarr = gdev(x.u[1])
    return xarr
end

#Build our over-all model topology
m = Lux.Chain(; down, nn_ode, convert = Lux.WrappedFunction(DiffEqArray_to_Array), fc)
ps, st = Lux.setup(Xoshiro(0), m)
ps = ComponentArray(ps) |> gdev
st = st |> gdev

#We can also build the model topology without a NN-ODE
m_no_ode = Lux.Chain(; down, nn, fc)
ps_no_ode, st_no_ode = Lux.setup(Xoshiro(0), m_no_ode)
ps_no_ode = ComponentArray(ps_no_ode) |> gdev
st_no_ode = st_no_ode |> gdev

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = first(down(x_train[1], ps.down, st.down))

# We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = first(m(x_train[1], ps, st))
#Or without the NN-ODE layer.
x_m = first(m_no_ode(x_train[1], ps_no_ode, st_no_ode))

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 100)
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
#burn in accuracy
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
        @info "[MNIST GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps, st))"
    end
    return false
end

# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); callback)
@test accuracy(m, zip(x_train, y_train), res.u, st) > 0.8
```

## Step-by-Step Description

### Load Packages

```@example mnist
using DiffEqFlux, CUDA, Zygote, MLDataUtils, NNlib, OrdinaryDiffEq, Test, Lux, Statistics,
      ComponentArrays, Random, Optimization, OptimizationOptimisers, LuxCUDA
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs
```

### GPU

A good trick used here:

```@example mnist
CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true
const cdev = cpu_device()
const gdev = gpu_device()
```

ensures that only optimized kernels are called when using the GPU.
Additionally, the `gpu_device` function is shown as a way to translate models and data over to the GPU.
Note that this function is CPU-safe, so if the GPU is disabled or unavailable, this
code will fall back to the CPU.

### Load MNIST Dataset into Minibatches

The MNIST dataset is split into 60,000 train and 10,000 test images, ensuring a balanced ratio of labels.

The preprocessing is done in `loadmnist` where the raw MNIST data is split into features `x` and labels `y`.
Features are reshaped into format **[Height, Width, Color, Samples]**, in case of the train set **[28, 28, 1, 60000]**.
Using MLDataUtils's `LabelEnc` function, the labels (numbers 0 to 9) are one-hot encoded, resulting in a a **[10, 60000]** `OneHotMatrix`.

Features and labels are then passed to MLDataUtils's `batchview`.
This automatically minibatches both the images and labels using the specified `batchsize`,
meaning that every minibatch will contain 128 images with a single color channel of 28x28 pixels.

```@example mnist
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
```

and then loaded from main:

```@example mnist
# Main
const bs = 128
x_train, y_train = loadmnist(bs)
```

### Layers

The Neural Network requires passing inputs sequentially through multiple layers. We use
`Chain` which allows inputs to functions to come from the previous layer and sends the outputs
to the next. Four different sets of layers are used here:

```@example mnist
down = Lux.Chain(Lux.FlattenLayer(), Lux.Dense(784, 20, tanh))
nn = Lux.Chain(Lux.Dense(20, 10, tanh), Lux.Dense(10, 10, tanh), Lux.Dense(10, 20, tanh))
fc = Lux.Dense(20, 10)
```

`down`: This layer downsamples our images into a 20 dimensional feature vector.
It takes a 28 x 28 image, flattens it, and then passes it through a fully connected
layer with `tanh` activation

`nn`: A 3 layers Deep Neural Network Chain with `tanh` activation which is used to model
our differential equation

`nn_ode`: ODE solver layer

`fc`: The final fully connected layer which maps our learned feature vector to the probability of
the feature vector of belonging to a particular class

### Array Conversion

When using `NeuralODE`, this function converts the ODESolution's `DiffEqArray` to
a Matrix (CuArray), and reduces the matrix from 3 to 2 dimensions for use in the next layer.

```@example mnist
nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

function DiffEqArray_to_Array(x)
    xarr = gdev(x.u[1])
    return xarr
end
```

For CPU: If this function does not automatically fall back to CPU when no GPU is present, we can
change `gdev(x)` to `Array(x)`.

### Build Topology

Next, we connect all layers together in a single chain:

```@example mnist
# Build our overall model topology
m = Lux.Chain(; down, nn_ode, convert = Lux.WrappedFunction(DiffEqArray_to_Array), fc)
ps, st = Lux.setup(Xoshiro(0), m)
ps = ComponentArray(ps) |> gdev
st = st |> gdev
```

### Prediction

To convert the classification back into readable numbers, we use `classify` which returns the
prediction by taking the arg max of the output for each column of the minibatch:

```@example mnist
classify(x) = argmax.(eachcol(x))
```

### Accuracy

We then evaluate the accuracy on `n_batches` at a time through the entire network:

```@example mnist
function accuracy(model, data, ps, st; n_batches = 100)
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
#burn in accuracy
accuracy(m, zip(x_train, y_train), ps, st)
```

### Training Parameters

Once we have our model, we can train our neural network by backpropagation using `Lux.train!`.
This function requires **Loss**, **Optimizer** and **Callback** functions.

#### Loss

**Cross Entropy** is the loss function computed here, which applies a **Softmax** operation on the
final output of our model. `logitcrossentropy` takes in the prediction from our
model `model(x)` and compares it to actual output `y`:

```@example mnist
function loss_function(ps, x, y)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

#burn in loss
loss_function(ps, x_train[1], y_train[1])
```

#### Optimizer

`Adam` is specified here as our optimizer with a **learning rate of 0.05**:

```@example mnist
opt = OptimizationOptimisers.Adam(0.05)
```

#### Callback

This callback function is used to print both the training and testing accuracy after
10 training iterations:

```@example mnist
iter = 0

opt_func = OptimizationFunction(
    (ps, _, x, y) -> loss_function(ps, x, y), Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps)

function callback(ps, l, pred)
    global iter += 1
    #Monitor that the weights do infact update
    #Every 10 training iterations show accuracy
    if (iter % 10 == 0)
        @info "[MNIST GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps, st))"
    end
    return false
end
```

### Train

To train our model, we select the appropriate trainable parameters of our network with `params`.
In our case, backpropagation is required for `down`, `nn_ode` and `fc`. Notice that the parameters
for Neural ODE is given by `nn_ode.p`:

```@example mnist
# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); callback)
acc = accuracy(m, zip(x_train, y_train), res.u, st)
@test acc > 0.8 # hide
acc # hide
```
