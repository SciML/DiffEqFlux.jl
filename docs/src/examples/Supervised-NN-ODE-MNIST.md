# GPU-based MNIST Neural ODE Classifier

Training a classifier for **MNIST** using a neural
ordinary differential equation on **GPUs** with **Minibatching**. (Step-by-step details below)

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, MLDataUtils, NNlib
using Flux: logitcrossentropy
using MLDatasets: MNIST
using CuArrays
CuArrays.allowscalar(false)


function loadmnist(batchsize = bs)
	# Use MLDataUtils LabelEnc for natural onehot conversion
  	onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	#Load MNIST
	imgs, labels_raw = MNIST.traindata();
	#Process images into (H,W,C,BS) batches
	x_train = Float32.(reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))) |> gpu
	x_train = batchview(x_train,batchsize)
	#Onehot and batch the labels
	y_train = onehot(labels_raw) |> gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end

#Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Chain(x->reshape(x,(28*28,:)),
             Dense(784,20,tanh)
            ) |> gpu

nn  = Chain(Dense(20,10,tanh),
            Dense(10,10,tanh),
            Dense(10,20,tanh)
           ) |> gpu


nn_ode = NeuralODE( nn, (0.f0, 1.f0), Tsit5(),
                        save_everystep = false,
                        reltol = 1e-3, abstol = 1e-3,
                        save_start = false ) |> gpu

fc  = Chain(Dense(20,10)) |> gpu

nfe = 0

function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end

#Build our over-all model topology
m        = Chain(down,
                 nn_ode,
                 DiffEqArray_to_Array,
                 fc) |> gpu

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = down(x_train[1])

#We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = m(x_train[1])

classify(x) = argmax.(eachcol(x))

function accuracy(model,data; n_batches=100)
    total_correct = 0
    total = 0
    for (x,y) in collect(data)[1:n_batches]
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct/total
end

#burn in accuracy
accuracy(m, zip(x_train,y_train))

loss(x,y) = logitcrossentropy(m(x),y)

#burn in loss
loss(x_train[1],y_train[1])

opt = ADAM(0.05)
iter = 0

cb() = begin
       global iter += 1
       #Monitor that the weights do infact update
       #Every 10 training iterations show accuracy
       (iter%10 == 0) && @show accuracy(m, zip(x_train,y_train))
       global nfe=0
end

# Train the NN-ODE and monitor the loss and weights.
Flux.train!( loss, params( down, nn_ode.p, fc), zip( x_train, y_train ), opt, cb = cb )
```


## Step-by-Step Description

### Load Packages

```julia
using DiffEqFlux, OrdinaryDiffEq, Flux, MLDataUtils, NNlib
using Flux: logitcrossentropy
using MLDatasets: MNIST
```

### GPU
A good trick used here:

```julia
using CuArrays
CuArrays.allowscalar(false)
```

ensures that only optimized kernels are called when using the GPU.
Additionally, the `gpu` function is shown as a way to translate models and data over to the GPU.
Note that this function is CPU-safe, so if the GPU is disabled or unavailable, this
code will fallback to the CPU. If the CPU fallback results in an error, we can also replace
gpu(...) with Array(...).

### Load MNIST Dataset into Minibatches

The preprocessing is done in `loadmnist` where the raw MNIST data is split into features `x_train`
and labels `y_train` by specifying batchsize `bs`. The function `convertlabel` will then transform
the current labels (`labels_raw`) from numbers 0 to 9 (`LabelEnc.NativeLabels(collect(0:9))`) into
one hot encoding (`LabelEnc.OneOfK`).

Features are reshaped into format **[Height, Width, Color, BatchSize]** or in this case **[28, 28, 1, 128]**
meaning that every minibatch will contain 128 images with a single color channel of 28x28 pixels.
The entire dataset of 60,000 images is then split into these batches with `batchview` resulting in
468 different batches (60,000 / 128) for both features and labels.

```julia
function loadmnist(batchsize = bs)
	#Use MLDataUtils LabelEnc for natural onehot conversion
  	onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	#Load MNIST
	imgs, labels_raw = MNIST.traindata();
	#Process images into (H,W,C,BS) batches
	x_train = Float32.(reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))) |> gpu
	x_train = batchview(x_train,batchsize)
	#Onehot and batch the labels
	y_train = onehot(labels_raw) |> gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end
```

and then loaded from main
```
#Main
const bs = 128
x_train, y_train = loadmnist(bs)
```


### Layers

The Neural Network requires passing inputs sequentially through multiple layers. We use
`Chain` which allows inputs to functions to come from previous layer and send the outputs
to the next. Four different sets of layers are used here.


```julia
down = Chain(x->reshape(x,(28*28,:)),
             Dense(784,20,tanh)
            ) |> gpu

nn  = Chain(Dense(20,10,tanh),
            Dense(10,10,tanh),
            Dense(10,20,tanh)
           ) |> gpu


nn_ode = NeuralODE(nn, (0.f0, 1.f0), Tsit5(),
                        save_everystep = false,
                        reltol = 1e-3, abstol = 1e-3,
                        save_start = false) |> gpu

fc  = Chain(Dense(20,10)) |> gpu

nfe = 0
```

`down`: Takes a 28x28 image, flattens it, then passes it through a fully connected
layer with `tanh` activation

`nn`: Neural network chain three layers deep with `tanh` activation

`nn_ode`: ODE solver layer

`fc`: Final fully connected layer

`nfe`: <mark><-- not sure </mark>

`|> gpu`: Sends to GPU

### Array Conversion

When using `NeuralODE`, we can use thie following function as a cheap conversion of `DiffEqArray`
from ODE solver into a Matrix that can be used for the following layer.

```julia
function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
end
```

### Build Topology

Next we connect all layers together in a single chain

```julia
#Build our over-all model topology
m = Chain(down,
          nn_ode,
          DiffEqArray_to_Array,
          fc) |> gpu

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = down(x_train[1])

#We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = m(x_train[1])
```

This can also be built without the NN-ODE by replacing nn-ode with a simple nn.

```julia
#We can also build the model topology without a NN-ODE
m_no_ode = Chain(down,
                 nn,
                 fc) |> gpu

x_m = m_no_ode(x_train[1])
```

### Prediction

To convert the classification back into a readable format, we use `classify` which returns the prediction by taking the arg max of the output for each column of the minibatch.

```julia
classify(x) = argmax.(eachcol(x))
```

### Accuracy

We then evaluate the accuracy on 100 batches through the entire network at a time.

```julia
function accuracy(model,data; n_batches=100)
    total_correct = 0
    total = 0
    for (x,y) in collect(data)[1:n_batches]
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct/total
end

#burn in accuracy
accuracy(m, zip(x_train,y_train))
```

### Loss
```julia
loss(x,y) = logitcrossentropy(m(x),y)

#burn in loss
loss(x_train[1],y_train[1])
```

### Optimizer
```julia
opt = ADAM(0.05)
iter = 0
```

### CallBack

```julia
cb() = begin
       global iter += 1
       #Monitor that the weights do infact update
       #Every 10 training iterations show accuracy
       (iter%10 == 0) && @show accuracy(m, zip(x_train,y_train))
       global nfe=0
end
```

### Train
```julia
# Train the NN-ODE and monitor the loss and weights.
Flux.train!(loss, params( down, nn_ode.p, fc), zip( x_train, y_train ), opt, cb = cb)
```

### Expected Output

```julia
accuracy(m, zip(x_train, y_train)) = 0.651015625
accuracy(m, zip(x_train, y_train)) = 0.72375
accuracy(m, zip(x_train, y_train)) = 0.77296875
accuracy(m, zip(x_train, y_train)) = 0.792421875
accuracy(m, zip(x_train, y_train)) = 0.82359375
accuracy(m, zip(x_train, y_train)) = 0.831875
accuracy(m, zip(x_train, y_train)) = 0.84203125
accuracy(m, zip(x_train, y_train)) = 0.826953125
accuracy(m, zip(x_train, y_train)) = 0.848515625
accuracy(m, zip(x_train, y_train)) = 0.784921875
accuracy(m, zip(x_train, y_train)) = 0.863203125
accuracy(m, zip(x_train, y_train)) = 0.8140625
accuracy(m, zip(x_train, y_train)) = 0.8596875
accuracy(m, zip(x_train, y_train)) = 0.86875
accuracy(m, zip(x_train, y_train)) = 0.865625
accuracy(m, zip(x_train, y_train)) = 0.874765625
accuracy(m, zip(x_train, y_train)) = 0.866640625
accuracy(m, zip(x_train, y_train)) = 0.867109375
accuracy(m, zip(x_train, y_train)) = 0.8746875
accuracy(m, zip(x_train, y_train)) = 0.8821875
accuracy(m, zip(x_train, y_train)) = 0.888125
accuracy(m, zip(x_train, y_train)) = 0.88421875
accuracy(m, zip(x_train, y_train)) = 0.88234375
accuracy(m, zip(x_train, y_train)) = 0.8775
accuracy(m, zip(x_train, y_train)) = 0.872890625
accuracy(m, zip(x_train, y_train)) = 0.868203125
accuracy(m, zip(x_train, y_train)) = 0.878359375
accuracy(m, zip(x_train, y_train)) = 0.869140625
accuracy(m, zip(x_train, y_train)) = 0.864296875
accuracy(m, zip(x_train, y_train)) = 0.84265625
accuracy(m, zip(x_train, y_train)) = 0.871953125
accuracy(m, zip(x_train, y_train)) = 0.8534375
accuracy(m, zip(x_train, y_train)) = 0.87859375
accuracy(m, zip(x_train, y_train)) = 0.859296875
accuracy(m, zip(x_train, y_train)) = 0.878515625
accuracy(m, zip(x_train, y_train)) = 0.884921875
accuracy(m, zip(x_train, y_train)) = 0.88296875
accuracy(m, zip(x_train, y_train)) = 0.880078125
accuracy(m, zip(x_train, y_train)) = 0.88765625
accuracy(m, zip(x_train, y_train)) = 0.88109375
accuracy(m, zip(x_train, y_train)) = 0.863515625
accuracy(m, zip(x_train, y_train)) = 0.869375
accuracy(m, zip(x_train, y_train)) = 0.87734375
accuracy(m, zip(x_train, y_train)) = 0.888515625
accuracy(m, zip(x_train, y_train)) = 0.88578125
accuracy(m, zip(x_train, y_train)) = 0.8790625
```
