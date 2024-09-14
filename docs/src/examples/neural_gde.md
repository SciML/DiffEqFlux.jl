# Neural Graph Differential Equations

!!! warn
    
    This tutorial has not been ran or updated in awhile.

This tutorial has been adapted from [here](https://github.com/CarloLucibello/GraphNeuralNetworks.jl/blob/master/examples/neural_ode_cora.jl).

In this tutorial, we will use Graph Differential Equations (GDEs) to perform classification on the [CORA Dataset](https://paperswithcode.com/dataset/cora). We shall be using the Graph Neural Networks primitives from the package [GraphNeuralNetworks](https://github.com/CarloLucibello/GraphNeuralNetworks.jl).

```julia
# Load the packages
using GraphNeuralNetworks, DifferentialEquations
using DiffEqFlux: NeuralODE
using GraphNeuralNetworks.GNNGraphs: normalized_adjacency
using Lux, NNlib, Optimisers, Zygote, Random, ComponentArrays
using Lux: AbstractLuxLayer, glorot_normal, zeros32
import Lux: initialparameters, initialstates
using SciMLSensitivity
using Statistics: mean
using MLDatasets: Cora
using CUDA
CUDA.allowscalar(false)
device = CUDA.functional() ? gpu : cpu

# Download the dataset
dataset = Cora();

# Preprocess the data and compute adjacency matrix
classes = dataset.metadata["classes"]
g = mldataset2gnngraph(dataset) |> device
onehotbatch(data, labels) = device(labels) .== reshape(data, 1, size(data)...)
onecold(y) = map(argmax, eachcol(y))
X = g.ndata.features
y = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal, but we don't want to use Flux here

Ã = normalized_adjacency(g; add_self_loops = true) |> device

(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:, train_mask]

# Model and Data Configuration
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 20

# Define the graph neural network
struct ExplicitGCNConv{F1, F2, F3, F4} <: AbstractLuxLayer
    in_chs::Int
    out_chs::Int
    activation::F1
    init_Ã::F2  # nomalized_adjacency matrix
    init_weight::F3
    init_bias::F4
end

function Base.show(io::IO, l::ExplicitGCNConv)
    print(io, "ExplicitGCNConv($(l.in_chs) => $(l.out_chs)")
    (l.activation == identity) || print(io, ", ", l.activation)
    print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv)
    return (weight = d.init_weight(rng, d.out_chs, d.in_chs),
        bias = d.init_bias(rng, d.out_chs, 1))
end

initialstates(rng::AbstractRNG, d::ExplicitGCNConv) = (Ã = d.init_Ã(),)

function ExplicitGCNConv(Ã, ch::Pair{Int, Int}, activation = identity;
        init_weight = glorot_normal, init_bias = zeros32)
    init_Ã = () -> copy(Ã)
    return ExplicitGCNConv{
        typeof(activation), typeof(init_Ã), typeof(init_weight), typeof(init_bias)}(
        first(ch), last(ch), activation, init_Ã, init_weight, init_bias)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x * st.Ã
    return l.activation.(z .+ ps.bias), st
end

# Define the Neural GDE
function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return dropdims(gpu(x); dims = 3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims = 3)

# make NeuralODE work with Lux.Chain
# remove this once https://github.com/SciML/DiffEqFlux.jl/issues/727 is fixed
initialparameters(rng::AbstractRNG, node::NeuralODE) = initialparameters(rng, node.model)
initialstates(rng::AbstractRNG, node::NeuralODE) = initialstates(rng, node.model)

gnn = Chain(ExplicitGCNConv(Ã, nhidden => nhidden, relu),
    ExplicitGCNConv(Ã, nhidden => nhidden, relu))

node = NeuralODE(gnn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

model = Chain(ExplicitGCNConv(Ã, nin => nhidden, relu),
    node, diffeqsol_to_array, Dense(nhidden, nout))

# Loss
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims = 1))

function loss(x, y, mask, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ[:, mask], y), st
end

function eval_loss_accuracy(X, y, mask, model, ps, st)
    ŷ, _ = model(X, ps, st)
    l = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l; digits = 4), acc = round(acc * 100; digits = 2))
end

# Training
function train()
    ## Setup model
    rng = Random.default_rng()
    Random.seed!(rng, 0)

    ps, st = Lux.setup(rng, model)
    ps = ComponentArray(ps) |> device
    st = st |> device

    ## Optimizer
    opt = Optimisers.Adam(0.01f0)
    st_opt = Optimisers.setup(opt, ps)

    ## Training Loop
    for _ in 1:epochs
        (l, st), back = pullback(p -> loss(X, ytrain, train_mask, model, p, st), ps)
        gs = back((one(l), nothing))[1]
        st_opt, ps = Optimisers.update(st_opt, ps, gs)
        @show eval_loss_accuracy(X, y, val_mask, model, ps, st)
    end
end

train()
```

# Step by Step Explanation

## Load the Required Packages

```julia
# Load the packages
using GraphNeuralNetworks, DifferentialEquations
using DiffEqFlux: NeuralODE
using GraphNeuralNetworks.GNNGraphs: normalized_adjacency
using Lux, NNlib, Optimisers, Zygote, Random, ComponentArrays
using Lux: AbstractLuxLayer, glorot_normal, zeros32
import Lux: initialparameters, initialstates
using SciMLSensitivity
using Statistics: mean
using MLDatasets: Cora
using CUDA
CUDA.allowscalar(false)
device = CUDA.functional() ? gpu : cpu
```

## Load the Dataset

The dataset is available in the desired format in the `MLDatasets` repository. We shall download the dataset from there.

```julia
dataset = Cora();
```

## Preprocessing the Data

Convert the data to `GNNGraph` and get the adjacency matrix from the graph `g`.

```julia
classes = dataset.metadata["classes"]
g = mldataset2gnngraph(dataset) |> device
onehotbatch(data, labels) = device(labels) .== reshape(data, 1, size(data)...)
onecold(y) = map(argmax, eachcol(y))
X = g.ndata.features
y = onehotbatch(g.ndata.targets, classes) # a dense matrix is not the optimal, but we don't want to use Flux here

Ã = normalized_adjacency(g; add_self_loops = true) |> device
```

### Training Data

GNNs operate on an entire graph, so we can't do any sort of minibatching here. We predict the entire dataset, but train the model in a semi-supervised learning fashion.

```julia
(; train_mask, val_mask, test_mask) = g.ndata
ytrain = y[:, train_mask]
```

## Model and Data Configuration

We shall use only 16 hidden state dimensions.

```julia
nin = size(X, 1)
nhidden = 16
nout = length(classes)
epochs = 20
```

## Define the Graph Neural Network

Here, we define a type of graph neural networks called `GCNConv`. We use the name `ExplicitGCNConv` to avoid naming conflicts with `GraphNeuralNetworks`. For more information on defining a layer with `Lux`, please consult to the [doc](http://lux.csail.mit.edu/dev/introduction/overview/#AbstractLuxLayer-API).

```julia
struct ExplicitGCNConv{F1, F2, F3} <: AbstractLuxLayer
    Ã::AbstractMatrix  # nomalized_adjacency matrix
    in_chs::Int
    out_chs::Int
    activation::F1
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, l::ExplicitGCNConv)
    print(io, "ExplicitGCNConv($(l.in_chs) => $(l.out_chs)")
    (l.activation == identity) || print(io, ", ", l.activation)
    print(io, ")")
end

function initialparameters(rng::AbstractRNG, d::ExplicitGCNConv)
    return (weight = d.init_weight(rng, d.out_chs, d.in_chs),
        bias = d.init_bias(rng, d.out_chs, 1))
end

function ExplicitGCNConv(Ã, ch::Pair{Int, Int}, activation = identity;
        init_weight = glorot_normal, init_bias = zeros32)
    return ExplicitGCNConv{typeof(activation), typeof(init_weight), typeof(init_bias)}(
        Ã, first(ch), last(ch), activation, init_weight, init_bias)
end

function (l::ExplicitGCNConv)(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x * l.Ã
    return l.activation.(z .+ ps.bias), st
end
```

## Neural Graph Ordinary Differential Equations

Let us now define the final model. We will use two GNN layers for approximating the gradients for the neural ODE. We use one additional `GCNConv` layer to project the data to a latent space and a `Dense` layer to project it from the latent space to the predictions. Finally, a softmax layer gives us the probability of the input belonging to each target category.

```julia
function diffeqsol_to_array(x::ODESolution{T, N, <:AbstractVector{<:CuArray}}) where {T, N}
    return dropdims(gpu(x); dims = 3)
end
diffeqsol_to_array(x::ODESolution) = dropdims(Array(x); dims = 3)

gnn = Chain(ExplicitGCNConv(Ã, nhidden => nhidden, relu),
    ExplicitGCNConv(Ã, nhidden => nhidden, relu))

node = NeuralODE(gnn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    reltol = 1e-3, abstol = 1e-3, save_start = false)

model = Chain(ExplicitGCNConv(Ã, nin => nhidden, relu),
    node, diffeqsol_to_array, Dense(nhidden, nout))
```

## Training Configuration

### Loss Function and Accuracy

We shall be using the standard categorical crossentropy loss function, which is used for multiclass classification tasks.

```julia
logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ); dims = 1))

function loss(x, y, mask, model, ps, st)
    ŷ, st = model(x, ps, st)
    return logitcrossentropy(ŷ[:, mask], y), st
end

function eval_loss_accuracy(X, y, mask, model, ps, st)
    ŷ, _ = model(X, ps, st)
    l = logitcrossentropy(ŷ[:, mask], y[:, mask])
    acc = mean(onecold(ŷ[:, mask]) .== onecold(y[:, mask]))
    return (loss = round(l; digits = 4), acc = round(acc * 100; digits = 2))
end
```

### Setup Model

We need to manually set up our mode with `Lux`, and convert the parameters to `ComponentArray` so that they can work well with sensitivity algorithms.

```julia
rng = Random.default_rng()
Random.seed!(rng, 0)

ps, st = Lux.setup(rng, model)
ps = ComponentArray(ps) |> device
st = st |> device
```

### Optimizer

For this task, we will be using the `Adam` optimizer with a learning rate of `0.01`.

```julia
opt = Optimisers.Adam(0.01f0)
st_opt = Optimisers.setup(opt, ps)
```

## Training Loop

Finally, we use the package `Optimisers` to learn the parameters `ps`. We run the training loop for `epochs` number of iterations.

```julia
for _ in 1:epochs
    (l, st), back = pullback(p -> loss(X, ytrain, train_mask, model, p, st), ps)
    gs = back((one(l), nothing))[1]
    st_opt, ps = Optimisers.update(st_opt, ps, gs)
    @show eval_loss_accuracy(X, y, val_mask, model, ps, st)
end
```
