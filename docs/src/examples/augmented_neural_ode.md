# Augmented Neural Ordinary Differential Equations

## Copy-Pasteable Code

```@example augneuralode_cp
using DiffEqFlux, OrdinaryDiffEq, Statistics, LinearAlgebra, Plots, LuxCUDA, Random
using MLUtils, ComponentArrays
using Optimization, OptimizationOptimisers, IterTools

const cdev = cpu_device()
const gdev = gpu_device()

function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(Float32, 1) .^ (1.0f0 / dim)) .+
               min_radius
    direction = randn(Float32, dim)
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
    data = cat(data...; dims = 2)
    labels = cat(labels...; dims = 2)
    return DataLoader((data |> gdev, labels |> gdev); batchsize = batch_size,
        shuffle = true, partial = false)
end

diffeqarray_to_array(x) = gdev(x.u[1])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(
        Chain(Dense(input_dim, hidden_dim, relu),
            Dense(hidden_dim, hidden_dim, relu), Dense(hidden_dim, input_dim)),
        (0.0f0, 1.0f0),
        Tsit5();
        save_everystep = false,
        reltol = 1.0f-3,
        abstol = 1.0f-3,
        save_start = false)
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    model = Chain(node, diffeqarray_to_array, Dense(input_dim, out_dim))
    ps, st = Lux.setup(Xoshiro(0), model)
    return model, ps |> gdev, st |> gdev
end

function plot_contour(model, ps, st, npoints = 300)
    grid_points = zeros(Float32, 2, npoints^2)
    idx = 1
    x = range(-4.0f0, 4.0f0; length = npoints)
    y = range(-4.0f0, 4.0f0; length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> gdev, ps, st)[1], npoints, npoints) |> cdev

    return contour(x, y, sol; fill = true, linewidth = 0.0)
end

loss_node(model, x, y, ps, st) = mean((first(model(x, ps, st)) .- y) .^ 2)

dataloader = concentric_sphere(
    2, (0.0f0, 2.0f0), (3.0f0, 4.0f0), 2000, 2000; batch_size = 256)

iter = 0
cb = function (ps, l)
    global iter
    iter += 1
    if iter % 10 == 0
        @info "Augmented Neural ODE" iter=iter loss=l
    end
    return false
end

model, ps, st = construct_model(1, 2, 64, 0)
opt = OptimizationOptimisers.Adam(0.005)

loss_node(model, dataloader.data[1], dataloader.data[2], ps, st)

println("Training Neural ODE")

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plt_node = plot_contour(model, res.u, st)

model, ps, st = construct_model(1, 2, 64, 1)
opt = OptimizationOptimisers.Adam(0.005)

println()
println("Training Augmented Neural ODE")

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plot_contour(model, res.u, st)
```

## Step-by-Step Explanation

### Loading required packages

```@example augneuralode
using DiffEqFlux, OrdinaryDiffEq, Statistics, LinearAlgebra, Plots, LuxCUDA, Random
using MLUtils, ComponentArrays
using Optimization, OptimizationOptimisers, IterTools

const cdev = cpu_device()
const gdev = gpu_device()
```

### Generating a toy dataset

In this example, we will be using data sampled uniformly in two concentric circles and then train our
Neural ODEs to do regression on that values. We assign `1` to any point which lies inside the inner
circle, and `-1` to any point which lies between the inner and outer circle. Our first function
`random_point_in_sphere` samples points uniformly between 2 concentric circles/spheres of radii
`min_radius` and `max_radius` respectively.

```@example augneuralode
function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(Float32, 1) .^ (1.0f0 / dim)) .+
               min_radius
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
    data = cat(data...; dims = 2)
    labels = cat(labels...; dims = 2)
    return DataLoader((data |> gdev, labels |> gdev); batchsize = batch_size,
        shuffle = true, partial = false)
end
```

### Models

We consider 2 models in this tutorial. The first is a simple Neural ODE which is described in detail in
[this tutorial](https://docs.sciml.ai/SciMLSensitivity/stable/examples/neural_ode/neural_ode_flux/). The other one is an
Augmented Neural ODE \[1]. The idea behind this layer is very simple. It augments the input to the Neural
DE Layer by appending zeros. So in order to use any arbitrary DE Layer in combination with this layer,
simply assume that the input to the DE Layer is of size `size(x, 1) + augment_dim` instead of `size(x, 1)`
and construct that layer accordingly.

In order to run the models on Flux.gpu, we need to manually transfer the models to Flux.gpu. First one is the network
predicting the derivatives inside the Neural ODE and the other one is the last layer in the Chain.

```@example augneuralode
diffeqarray_to_array(x) = gdev(x.u[1])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(
        Chain(Dense(input_dim, hidden_dim, relu),
            Dense(hidden_dim, hidden_dim, relu), Dense(hidden_dim, input_dim)),
        (0.0f0, 1.0f0),
        Tsit5();
        save_everystep = false,
        reltol = 1.0f-3,
        abstol = 1.0f-3,
        save_start = false)
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    model = Chain(node, diffeqarray_to_array, Dense(input_dim, out_dim))
    ps, st = Lux.setup(Xoshiro(0), model)
    return model, ps |> gdev, st |> gdev
end
```

### Plotting the Results

Here, we define a utility to plot our model regression results as a heatmap.

```@example augneuralode
function plot_contour(model, ps, st, npoints = 300)
    grid_points = zeros(Float32, 2, npoints^2)
    idx = 1
    x = range(-4.0f0, 4.0f0; length = npoints)
    y = range(-4.0f0, 4.0f0; length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points |> gdev, ps, st)[1], npoints, npoints) |> cdev

    return contour(x, y, sol; fill = true, linewidth = 0.0)
end
```

### Training Parameters

#### Loss Functions

We use the L2 distance between the model prediction `model(x)` and the actual prediction `y` as the
optimization objective.

```@example augneuralode
loss_node(model, x, y, ps, st) = mean((first(model(x, ps, st)) .- y) .^ 2)
```

#### Dataset

Next, we generate the dataset. We restrict ourselves to 2 dimensions as it is easy to visualize.
We sample a total of `4000` data points.

```@example augneuralode
dataloader = concentric_sphere(
    2, (0.0f0, 2.0f0), (3.0f0, 4.0f0), 2000, 2000; batch_size = 256)
```

#### Callback Function

Additionally, we define a callback function which displays the total loss at specific intervals.

```@example augneuralode
iter = 0
cb = function (ps, l)
    global iter
    iter += 1
    if iter % 10 == 0
        @info "Augmented Neural ODE" iter=iter loss=l
    end
    return false
end
```

#### Optimizer

We use Adam as the optimizer with a learning rate of 0.005

```@example augneuralode
opt = OptimizationOptimisers.Adam(5.0f-3)
```

### Training the Neural ODE

To train our neural ode model, we need to pass the appropriate learnable parameters, `parameters` which are
returned by the `construct_models` function. It is simply the `node.p` vector. We then train our model
for `20` epochs.

```@example augneuralode
model, ps, st = construct_model(1, 2, 64, 0)

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plot_contour(model, res.u, st)
```

Here is what the contour plot should look for Neural ODE. Notice that the regression is not perfect due to
the thin artifact which connects the circles.

## Training the Augmented Neural ODE

Our training configuration will be the same as that of Neural ODE. Only in this case, we have augmented the
input with a single zero. This makes the problem 3-dimensional, and as such it is possible to find
a function which can be expressed by the neural ode. For more details and proofs, please refer to [1].

```@example augneuralode
model, ps, st = construct_model(1, 2, 64, 1)

optfunc = OptimizationFunction(
    (x, p, data, target) -> loss_node(model, data, target, x, st),
    Optimization.AutoZygote())
optprob = OptimizationProblem(optfunc, ComponentArray(ps |> cdev) |> gdev)
res = solve(optprob, opt, IterTools.ncycle(dataloader, 5); callback = cb)

plot_contour(model, res.u, st)
```

## References

[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In Proceedings of the 33rd International Conference on Neural Information Processing Systems, pp. 3140-3150. 2019.
