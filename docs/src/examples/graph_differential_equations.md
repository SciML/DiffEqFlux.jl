# Neural Graph Differential Equations

This tutorial has been adapted from [here](https://github.com/yuehhua/GeometricFlux.jl/blob/master/examples/gcn.jl).

In this tutorial we will use Graph Differential Equations (GDEs) to perform classification on the [CORA Dataset](https://relational.fit.cvut.cz/dataset/CORA). We shall be using the Graph Neural Networks primitives from the package [GeometricFlux](https://github.com/yuehhua/GeometricFlux.jl).

**NOTE**: It is recommended to run this tutorial on CPU. The memory requirement of GPU for the training is > 32 GB

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
hidden = 256
target_catg = 7
epochs = 100

# Preprocess the data and compute adjacency matrix
train_X = Float32.(features)  # dim: num_features * num_nodes
train_y = Float32.(labels)  # dim: target_catg * num_nodes

adj_mat = Matrix{Float32}(adjacency_matrix(g))

# Define the Neural GDE
diffeqarray_to_array(x) = reshape(gpu(x), size(x)[1:2])

node = NeuralODE(
    Chain(GCNConv(adj_mat, hidden=>2hidden, tanh),
          GCNConv(adj_mat, 2hidden=>hidden)),
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

## Expected Output

```julia
accuracy(train_X, train_y) = 0.2182422451994091
accuracy(train_X, train_y) = 0.2311669128508124
accuracy(train_X, train_y) = 0.2488921713441654
accuracy(train_X, train_y) = 0.2780649926144756
accuracy(train_X, train_y) = 0.3338257016248153
accuracy(train_X, train_y) = 0.3611521418020679
accuracy(train_X, train_y) = 0.3947562776957164
accuracy(train_X, train_y) = 0.4198670605612998
accuracy(train_X, train_y) = 0.4326482301238421
accuracy(train_X, train_y) = 0.4578329149184101
accuracy(train_X, train_y) = 0.4678324123413124
accuracy(train_X, train_y) = 0.4989123389124312
accuracy(train_X, train_y) = 0.5195716395864106
accuracy(train_X, train_y) = 0.5217872968980798
accuracy(train_X, train_y) = 0.5243722304283605
accuracy(train_X, train_y) = 0.5258493353028065
accuracy(train_X, train_y) = 0.5280649926144756
accuracy(train_X, train_y) = 0.5284342688330872
accuracy(train_X, train_y) = 0.5284342688330872
accuracy(train_X, train_y) = 0.5295420974889217
accuracy(train_X, train_y) = 0.5324963072378139
accuracy(train_X, train_y) = 0.5321270310192023
accuracy(train_X, train_y) = 0.5336041358936484
accuracy(train_X, train_y) = 0.5328655834564254
accuracy(train_X, train_y) = 0.5343426883308715
accuracy(train_X, train_y) = 0.5354505169867061
accuracy(train_X, train_y) = 0.5369276218611522
accuracy(train_X, train_y) = 0.5384047267355982
accuracy(train_X, train_y) = 0.5391432791728212
accuracy(train_X, train_y) = 0.5413589364844904
accuracy(train_X, train_y) = 0.5409896602658789
accuracy(train_X, train_y) = 0.5420974889217134
accuracy(train_X, train_y) = 0.5439438700147711
accuracy(train_X, train_y) = 0.5450516986706057
accuracy(train_X, train_y) = 0.5465288035450517
accuracy(train_X, train_y) = 0.5480059084194978
accuracy(train_X, train_y) = 0.5487444608567208
accuracy(train_X, train_y) = 0.551698670605613
accuracy(train_X, train_y) = 0.5505908419497785
accuracy(train_X, train_y) = 0.551698670605613
accuracy(train_X, train_y) = 0.5539143279172821
accuracy(train_X, train_y) = 0.5542836041358936
accuracy(train_X, train_y) = 0.5564992614475628
accuracy(train_X, train_y) = 0.5564992614475628
accuracy(train_X, train_y) = 0.5576070901033974
accuracy(train_X, train_y) = 0.5564992614475628
accuracy(train_X, train_y) = 0.5579763663220089
accuracy(train_X, train_y) = 0.5557607090103397
accuracy(train_X, train_y) = 0.5568685376661743
accuracy(train_X, train_y) = 0.5594534711964549
accuracy(train_X, train_y) = 0.5583456425406204
accuracy(train_X, train_y) = 0.5583456425406204
accuracy(train_X, train_y) = 0.5590841949778435
accuracy(train_X, train_y) = 0.5605612998522895
accuracy(train_X, train_y) = 0.5624076809453471
accuracy(train_X, train_y) = 0.5642540620384048
accuracy(train_X, train_y) = 0.5642540620384048
accuracy(train_X, train_y) = 0.5649926144756278
accuracy(train_X, train_y) = 0.5653618906942393
accuracy(train_X, train_y) = 0.5657311669128509
accuracy(train_X, train_y) = 0.5672082717872969
accuracy(train_X, train_y) = 0.5690546528803545
accuracy(train_X, train_y) = 0.569423929098966
accuracy(train_X, train_y) = 0.5701624815361891
accuracy(train_X, train_y) = 0.5709010339734121
accuracy(train_X, train_y) = 0.5727474150664698
accuracy(train_X, train_y) = 0.5738552437223042
accuracy(train_X, train_y) = 0.5742245199409158
accuracy(train_X, train_y) = 0.5745937961595273
accuracy(train_X, train_y) = 0.5757016248153619
accuracy(train_X, train_y) = 0.5764401772525849
accuracy(train_X, train_y) = 0.5775480059084195
accuracy(train_X, train_y) = 0.5775480059084195
accuracy(train_X, train_y) = 0.5801329394387001
accuracy(train_X, train_y) = 0.579394387001477
accuracy(train_X, train_y) = 0.5819793205317577
accuracy(train_X, train_y) = 0.5827178729689808
accuracy(train_X, train_y) = 0.5823485967503693
accuracy(train_X, train_y) = 0.5838257016248154
accuracy(train_X, train_y) = 0.5827178729689808
accuracy(train_X, train_y) = 0.5827178729689808
accuracy(train_X, train_y) = 0.5823485967503693
accuracy(train_X, train_y) = 0.5819793205317577
accuracy(train_X, train_y) = 0.5823485967503693
accuracy(train_X, train_y) = 0.5816100443131462
accuracy(train_X, train_y) = 0.5812407680945347
accuracy(train_X, train_y) = 0.5823485967503693
accuracy(train_X, train_y) = 0.5867799113737076
accuracy(train_X, train_y) = 0.5867799113737076
accuracy(train_X, train_y) = 0.5864106351550961
accuracy(train_X, train_y) = 0.5823485967503693
accuracy(train_X, train_y) = 0.5827178729689808
accuracy(train_X, train_y) = 0.5830871491875923
accuracy(train_X, train_y) = 0.585672082717873
accuracy(train_X, train_y) = 0.5867799113737076
accuracy(train_X, train_y) = 0.587149187592319
accuracy(train_X, train_y) = 0.5853028064992615
accuracy(train_X, train_y) = 0.587149187592319
accuracy(train_X, train_y) = 0.5878877400295421
accuracy(train_X, train_y) = 0.5901033973412112
accuracy(train_X, train_y) = 0.5901033973412112
accuracy(train_X, train_y) = 0.5908419497784343
accuracy(train_X, train_y) = 0.5915805022156573
accuracy(train_X, train_y) = 0.5912112259970458
accuracy(train_X, train_y) = 0.5912112259970458
accuracy(train_X, train_y) = 0.5912112259970458
accuracy(train_X, train_y) = 0.5919497784342689
```