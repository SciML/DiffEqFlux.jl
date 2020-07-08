# Neural Graph Differential Equations

This tutorial has been adapted from [here](https://github.com/yuehhua/GeometricFlux.jl/blob/master/examples/gcn.jl).

In this tutorial we will use Graph Differential Equations (GDEs) to perform classification on the [CORA Dataset](https://relational.fit.cvut.cz/dataset/CORA). We shall be using the Graph Neural Networks primitives from the package [GeometricFlux](https://github.com/yuehhua/GeometricFlux.jl).

**NOTE**: It is recommended to run this tutorial on CPU. GCNConv uses scalar indexing which is slow on GPUs

```julia
# Load the packages
using GeometricFlux, Flux, JLD2, SparseArrays, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics: mean
using LightGraphs: adjacency_matrix

# Download the dataset
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_features.jld2", "cora_features.jld2")
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_graph.jld2", "cora_graph.jld2")
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_labels.jld2", "cora_labels.jld2")

# Load the dataset
@load "./cora_features.jld2" features
@load "./cora_labels.jld2" labels
@load "./cora_graph.jld2" g

# Model and Data Configuration
num_nodes = 2708
num_features = 1433
hidden = 16
target_catg = 7
epochs = 40

# Preprocess the data and compute adjacency matrix
train_X = Float32.(features)  # dim: num_features * num_nodes
train_y = Float32.(labels)  # dim: target_catg * num_nodes

adj_mat = Matrix{Float32}(adjacency_matrix(g))

# Define the Neural GDE
diffeqarray_to_array(x) = reshape(gpu(x), size(x)[1:2])

node = NeuralODE(
    GCNConv(adj_mat, hidden=>hidden),
    (0.f0, 1.f0), Tsit5(), save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false
)

model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              Dropout(0.5),
              node,
              diffeqarray_to_array,
              GCNConv(adj_mat, hidden=>target_catg),
              softmax) 

# Loss
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# Training
## Model Parameters
ps = Flux.params(model, node.p);

## Training Data
train_data = [(train_X, train_y)]

## Optimizer
opt = ADAM(0.01)

## Callback Function for printing accuracies
evalcb() = @show(accuracy(train_X, train_y))

## Training Loop
for i = 1:epochs
    Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
end
```

# Step by Step Explanation

## Load the Required Packages

```julia
# Load the packages
using GeometricFlux, Flux, JLD2, SparseArrays, DiffEqFlux, DifferentialEquations
using Flux: onehotbatch, onecold, crossentropy, throttle
using Statistics: mean
using LightGraphs: adjacency_matrix
```

## Load the Dataset

The dataset is available in the desired format in the GeometricFlux repository. We shall download the dataset from there, and use the JLD2 package to load the data.

```julia
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_features.jld2", "cora_features.jld2")
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_graph.jld2", "cora_graph.jld2")
download("https://rawcdn.githack.com/yuehhua/GeometricFlux.jl/a94ca7ce2ad01a12b23d68eb6cd991ee08569303/data/cora_labels.jld2", "cora_labels.jld2")

@load "./cora_features.jld2" features
@load "./cora_labels.jld2" labels
@load "./cora_graph.jld2" g
```

## Model and Data Configuration

The `num_nodes`, `target_catg` and `num_features` are defined by the data itself. We shall use a shallow GNN with only 16 hidden state dimension.

```julia
num_nodes = 2708
num_features = 1433
hidden = 16
target_catg = 7
epochs = 40
```

## Preprocessing the Data

Convert the data to float32 and use `LightGraphs` to get the adjacency matrix from the graph `g`.

```julia
train_X = Float32.(features)  # dim: num_features * num_nodes
train_y = Float32.(labels)  # dim: target_catg * num_nodes

adj_mat = Matrix{Float32}(adjacency_matrix(g))
```

## Neural Graph Ordinary Differential Equations

Let us now define the final model. We will use a single layer GNN for approximating the gradients for the neural ODE. We use two additional `GCNConv` layers, one to project the data to a latent space and the other to project it from the latent space to the predictions. Finally a softmax layer gives us the probability of the input belonging to each target category.

```julia
diffeqarray_to_array(x) = reshape(cpu(x), size(x)[1:2])

node = NeuralODE(
    GCNConv(adj_mat, hidden=>hidden),
    (0.f0, 1.f0), Tsit5(), save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false
)

model = Chain(GCNConv(adj_mat, num_features=>hidden, relu),
              Dropout(0.5),
              node,
              diffeqarray_to_array,
              GCNConv(adj_mat, hidden=>target_catg),
              softmax) 
```

## Training Configuration

### Loss Function and Accurary

We shall be using the standard categorical crossentropy loss function which is used for multiclass classification tasks.

```julia
loss(x, y) = crossentropy(model(x), y)
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))
```

### Model Parameters

Now we extract the model parameters which we want to learn.

```julia
ps = Flux.params(model, node.p);
```

### Training Data

GNNs operate on an entire graph, so we can't do any sort of minibatching here. We need to pass the entire data in a single pass. So our dataset is an array with a single tuple.

```julia
train_data = [(train_X, train_y)]
```

### Optimizer

For this task we will be using the `ADAM` optimizer with a learning rate of `0.01`.

```julia
opt = ADAM(0.01)
```

### Callback Function

We also define a utility function for printing the accuracy of the model over time.

```julia
evalcb() = @show(accuracy(train_X, train_y))
```

## Training Loop

Finally, with the configuration ready and all the utilities defined we can use the `Flux.train!` function to learn the parameters `ps`. We run the training loop for `epochs` number of iterations.

```julia
for i = 1:epochs
    Flux.train!(loss, ps, train_data, opt, cb=throttle(evalcb, 10))
end
```

## Expected Output

```julia
accuracy(train_X, train_y) = 0.12370753323485968
accuracy(train_X, train_y) = 0.11632200886262925
accuracy(train_X, train_y) = 0.1189069423929099
accuracy(train_X, train_y) = 0.13404726735598227
accuracy(train_X, train_y) = 0.15620384047267355
accuracy(train_X, train_y) = 0.1776218611521418
accuracy(train_X, train_y) = 0.19793205317577547
accuracy(train_X, train_y) = 0.21122599704579026
accuracy(train_X, train_y) = 0.22673559822747416
accuracy(train_X, train_y) = 0.2429837518463811
accuracy(train_X, train_y) = 0.25406203840472674
accuracy(train_X, train_y) = 0.26809453471196454
accuracy(train_X, train_y) = 0.2869276218611521
accuracy(train_X, train_y) = 0.2961595273264402
accuracy(train_X, train_y) = 0.30797636632200887
accuracy(train_X, train_y) = 0.31831610044313147
accuracy(train_X, train_y) = 0.3257016248153619
accuracy(train_X, train_y) = 0.3378877400295421
accuracy(train_X, train_y) = 0.3500738552437223
accuracy(train_X, train_y) = 0.3629985228951256
accuracy(train_X, train_y) = 0.37259970457902514
accuracy(train_X, train_y) = 0.3777695716395864
accuracy(train_X, train_y) = 0.3895864106351551
accuracy(train_X, train_y) = 0.396602658788774
accuracy(train_X, train_y) = 0.4010339734121123
accuracy(train_X, train_y) = 0.40472673559822747
accuracy(train_X, train_y) = 0.41285081240768096
accuracy(train_X, train_y) = 0.422821270310192
accuracy(train_X, train_y) = 0.43057607090103395
accuracy(train_X, train_y) = 0.43833087149187594
accuracy(train_X, train_y) = 0.44645494830132937
accuracy(train_X, train_y) = 0.4538404726735598
accuracy(train_X, train_y) = 0.45901033973412114
accuracy(train_X, train_y) = 0.4630723781388479
accuracy(train_X, train_y) = 0.46971935007385524
accuracy(train_X, train_y) = 0.474519940915805
accuracy(train_X, train_y) = 0.47858197932053176
accuracy(train_X, train_y) = 0.4815361890694239
accuracy(train_X, train_y) = 0.4804283604135894
accuracy(train_X, train_y) = 0.4848596750369276
```