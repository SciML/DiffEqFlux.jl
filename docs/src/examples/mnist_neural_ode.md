# GPU-based MNIST Neural ODE Classifier

Training a classifier for **MNIST** using a neural ordinary differential equation **NN-ODE**
on **GPUs** with **Minibatching**.

(Step-by-step description below)

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, NNlib, MLDataUtils, Printf
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
using CUDA
CUDA.allowscalar(false)

function loadmnist(batchsize = bs, train_split = 0.9)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata();
    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = stratifiedobs((x_data, y_data),
                                                         p = train_split)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(gpu.(collect.((x_train, y_train))); batchsize = batchsize,
                   shuffle = true),
        # Don't shuffle the test data
        DataLoader(gpu.(collect.((x_test, y_test))); batchsize = batchsize,
                   shuffle = false)
    )
end

# Main
const bs = 128
const train_split = 0.9
train_dataloader, test_dataloader = loadmnist(bs, train_split)

down = Chain(flatten, Dense(784, 20, tanh)) |> gpu

nn = Chain(Dense(20, 10, tanh),
           Dense(10, 10, tanh),
           Dense(10, 20, tanh)) |> gpu


nn_ode = NeuralODE(nn, (0.f0, 1.f0), Tsit5(),
                   save_everystep = false,
                   reltol = 1e-3, abstol = 1e-3,
                   save_start = false) |> gpu

fc  = Chain(Dense(20, 10)) |> gpu

function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

# Build our over-all model topology
model = Chain(down,
              nn_ode,
              DiffEqArray_to_Array,
              fc) |> gpu;

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]

x_d = down(img)

# We can see that we can compute the forward pass through the NN topology
# featuring an NNODE layer.
x_m = model(img)

classify(x) = argmax.(eachcol(x))

function accuracy(model, data; n_batches = 100)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        # Only evaluate accuracy for n_batches
        i > n_batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(model, train_dataloader)

loss(x, y) = logitcrossentropy(model(x), y)

# burn in loss
loss(img, lab)

opt = ADAM(0.05)
iter = 0

cb() = begin
    global iter += 1
    # Monitor that the weights do infact update
    # Every 10 training iterations show accuracy
    if iter % 10 == 1
        train_accuracy = accuracy(model, train_dataloader) * 100
        test_accuracy = accuracy(model, test_dataloader;
                                 n_batches = length(test_dataloader)) * 100
        @printf("Iter: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
                iter, train_accuracy, test_accuracy)
    end
end

# Train the NN-ODE and monitor the loss and weights.
Flux.train!(loss, params(down, nn_ode.p, fc), train_dataloader, opt, cb = cb)
```


## Step-by-Step Description

### Load Packages

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, NNlib, MLDataUtils, Printf
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets
```

### GPU
A good trick used here:

```julia
using CUDA
CUDA.allowscalar(false)
```

ensures that only optimized kernels are called when using the GPU.
Additionally, the `gpu` function is shown as a way to translate models and data over to the GPU.
Note that this function is CPU-safe, so if the GPU is disabled or unavailable, this
code will fallback to the CPU.

### Load MNIST Dataset into Minibatches

The preprocessing is done in `loadmnist` where the raw MNIST data is split into features `x_train`
and labels `y_train` by specifying batchsize `bs`. The function `convertlabel` will then transform
the current labels (`labels_raw`) from numbers 0 to 9 (`LabelEnc.NativeLabels(collect(0:9))`) into
one hot encoding (`LabelEnc.OneOfK`).

Features are reshaped into format **[Height, Width, Color, BatchSize]** or in this case **[28, 28, 1, 128]**
meaning that every minibatch will contain 128 images with a single color channel of 28x28 pixels.
The entire dataset of 60,000 images is split into the train and test dataset, ensuring a balanced ratio
of labels. These splits are then passed to Flux's DataLoader. This automatically minibatches both the images and
labels. Additionally, it allows us to shuffle the train dataset in each epoch while keeping the order of the
test data the same.

```julia
function loadmnist(batchsize = bs, train_split = 0.9)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata();
    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = stratifiedobs((x_data, y_data),
                                                         p = train_split)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(gpu.(collect.((x_train, y_train))); batchsize = batchsize,
                   shuffle = true),
        # Don't shuffle the test data
        DataLoader(gpu.(collect.((x_test, y_test))); batchsize = batchsize,
                   shuffle = false)
    )
end
```

and then loaded from main:
```
# Main
const bs = 128
const train_split = 0.9
train_dataloader, test_dataloader = loadmnist(bs, train_split)
```


### Layers

The Neural Network requires passing inputs sequentially through multiple layers. We use
`Chain` which allows inputs to functions to come from previous layer and sends the outputs
to the next. Four different sets of layers are used here:


```julia
down = Chain(flatten, Dense(784, 20, tanh)) |> gpu

nn = Chain(Dense(20, 10, tanh),
           Dense(10, 10, tanh),
           Dense(10, 20, tanh)) |> gpu


nn_ode = NeuralODE(nn, (0.f0, 1.f0), Tsit5(),
                   save_everystep = false,
                   reltol = 1e-3, abstol = 1e-3,
                   save_start = false) |> gpu

fc  = Chain(Dense(20, 10)) |> gpu
```

`down`: This layer downsamples our images into a 20 dimensional feature vector.
        It takes a 28 x 28 image, flattens it, and then passes it through a fully connected
        layer with `tanh` activation

`nn`: A 3 layers Deep Neural Network Chain with `tanh` activation which is used to model
      our differential equation

`nn_ode`: ODE solver layer

`fc`: The final fully connected layer which maps our learned feature vector to the probability of
      the feature vector of belonging to a particular class

`|> gpu`: An utility function which transfers our model to GPU, if it is available

### Array Conversion

When using `NeuralODE`, we can use the following function as a cheap conversion of `DiffEqArray`
from the ODE solver into a Matrix that can be used in the following layer:

```julia
function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end
```

For CPU: If this function does not automatically fallback to CPU when no GPU is present, we can
change `gpu(x)` with `Array(x)`.


### Build Topology

Next we connect all layers together in a single chain:

```julia
# Build our over-all model topology
model = Chain(down,
              nn_ode,
              DiffEqArray_to_Array,
              fc) |> gpu;
```

There are a few things we can do to examine the inner workings of our neural network:

```julia
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = down(img)

# We can see that we can compute the forward pass through the NN topology
# featuring an NNODE layer.
x_m = model(img)
```

This can also be built without the NN-ODE by replacing `nn-ode` with a simple `nn`:

```julia
# We can also build the model topology without a NN-ODE
m_no_ode = Chain(down,
                 nn,
                 fc) |> gpu

x_m = m_no_ode(img)
```

### Prediction

To convert the classification back into readable numbers, we use `classify` which returns the
prediction by taking the arg max of the output for each column of the minibatch:

```julia
classify(x) = argmax.(eachcol(x))
```

### Accuracy

We then evaluate the accuracy on `n_batches` at a time through the entire network:

```julia
function accuracy(model, data; n_batches = 100)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(collect(data))
        # Only evaluate accuracy for n_batches
        i > n_batches && break
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

# burn in accuracy
accuracy(m, train_dataloader)
```

### Training Parameters

Once we have our model, we can train our neural network by backpropagation using `Flux.train!`.
This function requires **Loss**, **Optimizer** and **Callback** functions.

#### Loss

**Cross Entropy** is the loss function computed here which applies a **Softmax** operation on the
final output of our model. `logitcrossentropy` takes in the prediction from our
model `model(x)` and compares it to actual output `y`:

```julia
loss(x, y) = logitcrossentropy(model(x), y)

# burn in loss
loss(img, lab)
```

#### Optimizer

`ADAM` is specified here as our optimizer with a **learning rate of 0.05**:

```julia
opt = ADAM(0.05)
```

#### CallBack

This callback function is used to print both the training and testing accuracy after
10 training iterations:

```julia
cb() = begin
    global iter += 1
    # Monitor that the weights do infact update
    # Every 10 training iterations show accuracy
    if iter % 10 == 1
        train_accuracy = accuracy(model, train_dataloader) * 100
        test_accuracy = accuracy(model, test_dataloader;
                                 n_batches = length(test_dataloader)) * 100
        @printf("Iter: %3d || Train Accuracy: %2.3f || Test Accuracy: %2.3f\n",
                iter, train_accuracy, test_accuracy)
    end
end
```

### Train

To train our model, we select the appropriate trainable parameters of our network with `params`.
In our case, backpropagation is required for `down`, `nn_ode` and `fc`. Notice that the parameters
for Neural ODE is given by `nn_ode.p`:

```julia
# Train the NN-ODE and monitor the loss and weights.
Flux.train!(loss, params( down, nn_ode.p, fc), zip( x_train, y_train ), opt, cb = cb)
```

### Expected Output

```julia
Iter:   1 || Train Accuracy: 16.203 || Test Accuracy: 16.933
Iter:  11 || Train Accuracy: 64.406 || Test Accuracy: 64.900
Iter:  21 || Train Accuracy: 76.656 || Test Accuracy: 76.667
Iter:  31 || Train Accuracy: 81.758 || Test Accuracy: 81.683
Iter:  41 || Train Accuracy: 81.078 || Test Accuracy: 81.967
Iter:  51 || Train Accuracy: 83.953 || Test Accuracy: 84.417
Iter:  61 || Train Accuracy: 85.266 || Test Accuracy: 85.017
Iter:  71 || Train Accuracy: 85.938 || Test Accuracy: 86.400
Iter:  81 || Train Accuracy: 84.836 || Test Accuracy: 85.533
Iter:  91 || Train Accuracy: 86.148 || Test Accuracy: 86.583
Iter: 101 || Train Accuracy: 83.859 || Test Accuracy: 84.500
Iter: 111 || Train Accuracy: 86.227 || Test Accuracy: 86.617
Iter: 121 || Train Accuracy: 87.508 || Test Accuracy: 87.200
Iter: 131 || Train Accuracy: 86.227 || Test Accuracy: 85.917
Iter: 141 || Train Accuracy: 84.453 || Test Accuracy: 84.850
Iter: 151 || Train Accuracy: 86.063 || Test Accuracy: 85.650
Iter: 161 || Train Accuracy: 88.375 || Test Accuracy: 88.033
Iter: 171 || Train Accuracy: 87.398 || Test Accuracy: 87.683
Iter: 181 || Train Accuracy: 88.070 || Test Accuracy: 88.350
Iter: 191 || Train Accuracy: 86.836 || Test Accuracy: 87.150
Iter: 201 || Train Accuracy: 89.266 || Test Accuracy: 88.583
Iter: 211 || Train Accuracy: 86.633 || Test Accuracy: 85.550
Iter: 221 || Train Accuracy: 89.313 || Test Accuracy: 88.217
Iter: 231 || Train Accuracy: 88.641 || Test Accuracy: 89.417
Iter: 241 || Train Accuracy: 88.617 || Test Accuracy: 88.550
Iter: 251 || Train Accuracy: 88.211 || Test Accuracy: 87.950
Iter: 261 || Train Accuracy: 87.742 || Test Accuracy: 87.317
Iter: 271 || Train Accuracy: 89.070 || Test Accuracy: 89.217
Iter: 281 || Train Accuracy: 89.703 || Test Accuracy: 89.067
Iter: 291 || Train Accuracy: 88.484 || Test Accuracy: 88.250
Iter: 301 || Train Accuracy: 87.898 || Test Accuracy: 88.367
Iter: 311 || Train Accuracy: 88.438 || Test Accuracy: 88.633
Iter: 321 || Train Accuracy: 88.664 || Test Accuracy: 88.567
Iter: 331 || Train Accuracy: 89.906 || Test Accuracy: 89.883
Iter: 341 || Train Accuracy: 88.883 || Test Accuracy: 88.667
Iter: 351 || Train Accuracy: 89.609 || Test Accuracy: 89.283
Iter: 361 || Train Accuracy: 89.516 || Test Accuracy: 89.117
Iter: 371 || Train Accuracy: 89.898 || Test Accuracy: 89.633
Iter: 381 || Train Accuracy: 89.055 || Test Accuracy: 89.017
Iter: 391 || Train Accuracy: 89.445 || Test Accuracy: 89.467
Iter: 401 || Train Accuracy: 89.156 || Test Accuracy: 88.250
Iter: 411 || Train Accuracy: 88.977 || Test Accuracy: 89.083
Iter: 421 || Train Accuracy: 90.109 || Test Accuracy: 89.417
```
