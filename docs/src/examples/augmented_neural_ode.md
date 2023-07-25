# Augmented Neural Ordinary Differential Equations

## Copy-Pasteable Code

```@example augneuralode_cp
using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
using Flux.Data: DataLoader

function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(Float32,1) .^ (1f0 / dim)) .+ min_radius
    direction = randn(Float32,dim)
    unit_direction = direction ./ norm(direction)
    return distance .* unit_direction
end

function concentric_sphere(dim, inner_radius_range, outer_radius_range,
                           num_samples_inner, num_samples_outer; batch_size = 64)
    data = []
    labels = []
    for _ in 1:num_samples_inner
        push!(data, reshape(random_point_in_sphere(dim, inner_radius_range...), :, 1))
        push!(labels, ones(1, 1))
    end
    for _ in 1:num_samples_outer
        push!(data, reshape(random_point_in_sphere(dim, outer_radius_range...), :, 1))
        push!(labels, -ones(1, 1))
    end
    data = cat(data..., dims=2)
    labels = cat(labels..., dims=2)
    DataLoader((data |> Flux.gpu, labels |> Flux.gpu); batchsize=batch_size, shuffle=true,
                      partial=false)
end

diffeqarray_to_array(x) = reshape(Flux.gpu(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(Flux.Chain(Flux.Dense(input_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, input_dim)) |> Flux.gpu,
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1f-3, abstol = 1f-3, save_start = false) |> Flux.gpu
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    return Flux.Chain((x, p=node.p) -> node(x, p),
                 Array,
                 diffeqarray_to_array,
                 Flux.Dense(input_dim, out_dim) |> Flux.gpu), node.p |> Flux.gpu
end

function plot_contour(model, npoints = 300)
    grid_points = zeros(Float32, 2, npoints ^ 2)
    idx = 1
    x = range(-4f0, 4f0, length = npoints)
    y = range(-4f0, 4f0, length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> Flux.gpu), npoints, npoints) |> Flux.cpu

    return contour(x, y, sol, fill = true, linewidth=0.0)
end

loss_node(x, y) = mean((model(x) .- y) .^ 2)

println("Generating Dataset")

dataloader = concentric_sphere(2, (0f0, 2f0), (3f0, 4f0), 2000, 2000; batch_size = 256)

iter = 0
cb = function()
    global iter 
    iter += 1
    if iter % 10 == 0
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end

model, parameters = construct_model(1, 2, 64, 0)
opt = ADAM(0.005)

println("Training Neural ODE")

for _ in 1:10
    Flux.train!(loss_node, Flux.params(parameters, model), dataloader, opt, cb = cb)
end

plt_node = plot_contour(model)

model, parameters = construct_model(1, 2, 64, 1)
opt = ADAM(5f-3)

println()
println("Training Augmented Neural ODE")

for _ in 1:10
    Flux.train!(loss_node, Flux.params(parameters, model), dataloader, opt, cb = cb)
end

plt_anode = plot_contour(model)
```

# Step-by-Step Explanation

## Loading required packages

```@example augneuralode
using DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
using Flux.Data: DataLoader
```

## Generating a toy dataset

In this example, we will be using data sampled uniformly in two concentric circles and then train our
Neural ODEs to do regression on that values. We assign `1` to any point which lies inside the inner
circle, and `-1` to any point which lies between the inner and outer circle. Our first function
`random_point_in_sphere` samples points uniformly between 2 concentric circles/spheres of radii
`min_radius` and `max_radius` respectively.

```@example augneuralode
function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(Float32, 1) .^ (1f0 / dim)) .+ min_radius
    direction = randn(Float32, dim)
    unit_direction = direction ./ norm(direction)
    return distance .* unit_direction
end
```

Next, we will construct a dataset of these points and use Flux's DataLoader to automatically minibatch
and shuffle the data.

```@example augneuralode
function concentric_sphere(dim, inner_radius_range, outer_radius_range,
                           num_samples_inner, num_samples_outer; batch_size = 64)
    data = []
    labels = []
    for _ in 1:num_samples_inner
        push!(data, reshape(random_point_in_sphere(dim, inner_radius_range...), :, 1))
        push!(labels, ones(1, 1))
    end
    for _ in 1:num_samples_outer
        push!(data, reshape(random_point_in_sphere(dim, outer_radius_range...), :, 1))
        push!(labels, -ones(1, 1))
    end
    data = cat(data..., dims=2)
    labels = cat(labels..., dims=2)
    return DataLoader((data |> Flux.gpu, labels |> Flux.gpu); batchsize=batch_size, shuffle=true,
                      partial=false)
end
```

## Models

We consider 2 models in this tutorial. The first is a simple Neural ODE which is described in detail in
[this tutorial](https://docs.sciml.ai/SciMLSensitivity/stable/neural_ode/neural_ode_flux/). The other one is an
Augmented Neural ODE \[1]. The idea behind this layer is very simple. It augments the input to the Neural
DE Layer by appending zeros. So in order to use any arbitrary DE Layer in combination with this layer,
simply assume that the input to the DE Layer is of size `size(x, 1) + augment_dim` instead of `size(x, 1)`
and construct that layer accordingly.

In order to run the models on Flux.gpu, we need to manually transfer the models to Flux.gpu. First one is the network
predicting the derivatives inside the Neural ODE and the other one is the last layer in the Chain.

```@example augneuralode
diffeqarray_to_array(x) = reshape(Flux.gpu(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(Flux.Chain(Flux.Dense(input_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, hidden_dim, relu),
                           Flux.Dense(hidden_dim, input_dim)) |> Flux.gpu,
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1f-3, abstol = 1f-3, save_start = false) |> Flux.gpu
    node = augment_dim == 0 ? node : (AugmentedNDELayer(node, augment_dim) |> Flux.gpu)
    return Flux.Chain((x, p=node.p) -> node(x, p),
                 Array,
                 diffeqarray_to_array,
                 Flux.Dense(input_dim, out_dim) |> Flux.gpu), node.p |> Flux.gpu
end
```

## Plotting the Results

Here, we define a utility to plot our model regression results as a heatmap.

```@example augneuralode
function plot_contour(model, npoints = 300)
    grid_points = zeros(2, npoints ^ 2)
    idx = 1
    x = range(-4f0, 4f0, length = npoints)
    y = range(-4f0, 4f0, length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> Flux.gpu), npoints, npoints) |> Flux.cpu

    return contour(x, y, sol, fill = true, linewidth=0.0)
end
```

## Training Parameters

### Loss Functions

We use the L2 distance between the model prediction `model(x)` and the actual prediction `y` as the
optimization objective.

```@example augneuralode
loss_node(x, y) = mean((model(x) .- y) .^ 2)
```

### Dataset

Next, we generate the dataset. We restrict ourselves to 2 dimensions as it is easy to visualize.
We sample a total of `4000` data points.

```@example augneuralode
dataloader = concentric_sphere(2, (0f0, 2f0), (3f0, 4f0), 2000, 2000; batch_size = 256)
```

### Callback Function

Additionally, we define a callback function which displays the total loss at specific intervals.

```@example augneuralode
iter = 0
cb = function()
    global iter += 1
    if iter % 10 == 1
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end
```

### Optimizer

We use ADAM as the optimizer with a learning rate of 0.005

```@example augneuralode
opt = ADAM(5f-3)
```

## Training the Neural ODE

To train our neural ode model, we need to pass the appropriate learnable parameters, `parameters` which are
returned by the `construct_models` function. It is simply the `node.p` vector. We then train our model
for `20` epochs.

```@example augneuralode
model, parameters = construct_model(1, 2, 64, 0)

for _ in 1:10
    Flux.train!(loss_node, Flux.params(model, parameters), dataloader, opt, cb = cb)
end
```

Here is what the contour plot should look for Neural ODE. Notice that the regression is not perfect due to
the thin artifact which connects the circles.

![node](https://user-images.githubusercontent.com/30564094/85916605-00f31500-b870-11ea-9857-5bf1f8c0477f.png)

## Training the Augmented Neural ODE

Our training configuration will be the same as that of Neural ODE. Only in this case, we have augmented the
input with a single zero. This makes the problem 3-dimensional, and as such it is possible to find
a function which can be expressed by the neural ode. For more details and proofs, please refer to [1].

```@example augneuralode
model, parameters = construct_model(1, 2, 64, 1)

for _ in 1:10
    Flux.train!(loss_node, Flux.params(model, parameters), dataloader, opt, cb = cb)
end
```

For the augmented Neural ODE we notice that the artifact is gone.

![anode](https://user-images.githubusercontent.com/30564094/85916607-02bcd880-b870-11ea-84fa-d15e24295ea6.png)

# Expected Output

```
Generating Dataset
Training Neural ODE
Iteration 10 || Loss = 0.9802582
Iteration 20 || Loss = 0.6727416
Iteration 30 || Loss = 0.5862373
Iteration 40 || Loss = 0.5278132
Iteration 50 || Loss = 0.4867624
Iteration 60 || Loss = 0.41630346
Iteration 70 || Loss = 0.3325938
Iteration 80 || Loss = 0.28235924
Iteration 90 || Loss = 0.24069068
Iteration 100 || Loss = 0.20503852
Iteration 110 || Loss = 0.17608969
Iteration 120 || Loss = 0.1491399
Iteration 130 || Loss = 0.12711425
Iteration 140 || Loss = 0.10686825
Iteration 150 || Loss = 0.089558244

Training Augmented Neural ODE
Iteration 10 || Loss = 1.3911372
Iteration 20 || Loss = 0.7694144
Iteration 30 || Loss = 0.5639633
Iteration 40 || Loss = 0.33187616
Iteration 50 || Loss = 0.14787851
Iteration 60 || Loss = 0.094676435
Iteration 70 || Loss = 0.07363529
Iteration 80 || Loss = 0.060333826
Iteration 90 || Loss = 0.04998395
Iteration 100 || Loss = 0.044843454
Iteration 110 || Loss = 0.042587914
Iteration 120 || Loss = 0.042706195
Iteration 130 || Loss = 0.040252227
Iteration 140 || Loss = 0.037686247
Iteration 150 || Loss = 0.036247417
```

# References

[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3140-3150. 2019.

