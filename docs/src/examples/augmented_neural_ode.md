# Augmented Neural Ordinary Differential Equations

```julia
using Flux, DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
import Flux.Data: DataLoader

function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(1) .^ (1.0 / dim)) .+ min_radius
    direction = randn(dim)
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
    return DataLoader(data, labels; batchsize=batch_size, shuffle=true, partial=false)
end

diffeqarray_to_array(x) = reshape(Array(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(FastChain(FastDense(input_dim, hidden_dim, relu),
                               FastDense(hidden_dim, hidden_dim, relu),
                               FastDense(hidden_dim, input_dim)),
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1e-3, abstol = 1e-3, save_start = false)
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    return Chain((x, p=node.p) -> node(x, p),
                 diffeqarray_to_array,
                 Dense(input_dim, out_dim)), node.p
end

function plot_contour(model, npoints = 300)
    grid_points = zeros(2, npoints ^ 2)
    idx = 1
    x = range(-4.0, 4.0, length = npoints)
    y = range(-4.0, 4.0, length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points), npoints, npoints)
    
    return contour(x, y, sol, fill = true, linewidth=0.0)
end

loss_node(x, y) = mean((model(x) .- y) .^ 2)

println("Generating Dataset")

dataloader = concentric_sphere(2, (0.0, 2.0), (3.0, 4.0), 2000, 2000; batch_size = 256)

cb = function()
    global iter += 1
    if iter % 10 == 0
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end

model, parameters = construct_model(1, 2, 64, 0)
opt = ADAM(0.005)
iter = 0

println("Training Neural ODE")

for _ in 1:20
    Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)
end

plt_node = plot_contour(model)

model, parameters = construct_model(1, 2, 64, 1)
opt = ADAM(0.005)
iter = 0

println()
println("Training Augmented Neural ODE")

for _ in 1:20
    Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)
end

plt_anode = plot_contour(model)
```

# Step-by-Step Explaination

## Loading required packages

```julia
using Flux, DiffEqFlux, DifferentialEquations
using Statistics, LinearAlgebra, Plots
import Flux.Data: DataLoader
```

## Generating a toy dataset

In this example, we will be using data sampled uniformly in two concentric circles and then train our
Neural ODEs to do regression on that values. We assign `1` to any point which lies inside the inner
circle, and `-1` to any point which lies between the inner and outer circle. Our first function
`random_point_in_sphere` samples points uniformly between 2 concentric circles/spheres of radii
`min_radius` and `max_radius` respectively.

```julia
function random_point_in_sphere(dim, min_radius, max_radius)
    distance = (max_radius - min_radius) .* (rand(1) .^ (1.0 / dim)) .+ min_radius
    direction = randn(dim)
    unit_direction = direction ./ norm(direction)
    return distance .* unit_direction
end
```

Next, we will construct a dataset of these points and use Flux's DataLoader to automatically minibatch
and shuffle the data.

```julia
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
    return DataLoader(data, labels; batchsize=batch_size, shuffle=true, partial=false)
end
```

## Models

We consider 2 models in this tuturial. The first is a simple Neural ODE which is described in detail in
[this tutorial](https://diffeqflux.sciml.ai/dev/examples/neural_ode_sciml/). The other one is an
Augmented Neural ODE \[1]. The idea behind this layer is very simple. It augments the input to the Neural
DE Layer by appending zeros. So in order to use any arbitrary DE Layer in combination with this layer,
simply assume that the input to the DE Layer is of size `size(x, 1) + augment_dim` instead of `size(x, 1)`
and construct that layer accordingly.

```julia
diffeqarray_to_array(x) = reshape(Array(x), size(x)[1:2])

function construct_model(out_dim, input_dim, hidden_dim, augment_dim)
    input_dim = input_dim + augment_dim
    node = NeuralODE(FastChain(FastDense(input_dim, hidden_dim, relu),
                               FastDense(hidden_dim, hidden_dim, relu),
                               FastDense(hidden_dim, input_dim)),
                     (0.f0, 1.f0), Tsit5(), save_everystep = false,
                     reltol = 1e-3, abstol = 1e-3, save_start = false)
    node = augment_dim == 0 ? node : AugmentedNDELayer(node, augment_dim)
    return Chain((x, p=node.p) -> node(x, p),
                 diffeqarray_to_array,
                 Dense(input_dim, out_dim)), node.p
end
```

## Plotting the Results

Here, we define an utility to plot our model regression results as a heatmap.

```julia
function plot_contour(model, npoints = 300)
    grid_points = zeros(2, npoints ^ 2)
    idx = 1
    x = range(-4.0, 4.0, length = npoints)
    y = range(-4.0, 4.0, length = npoints)
    for x1 in x, x2 in y
        grid_points[:, idx] .= [x1, x2]
        idx += 1
    end
    sol = reshape(model(grid_points), npoints, npoints)
    
    return contour(x, y, sol, fill = true, linewidth=0.0)
end
```

## Training Parameters

### Loss Functions

We use the L2 distance between the model prediction `model(x)` and the actual prediction `y` as the
optimization objective.

```julia
loss_node(x, y) = mean((model(x) .- y) .^ 2)
```

### Dataset

Next, we generate the dataset. We restrict ourselves to 2 dimensions as it is easy to visualize.
We sample a total of `4000` data points.

```julia
dataloader = concentric_sphere(2, (0.0, 2.0), (3.0, 4.0), 2000, 2000; batch_size = 256)
```

### Callback Function

Additionally we define a callback function which displays the total loss at specific intervals.

```julia
cb = function()
    global iter += 1
    if iter % 10 == 1
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end
```

### Optimizer

We use ADAM as the optimizer with a learning rate of 0.005

```julia
opt = ADAM(0.005)
```

## Training the Neural ODE

To train our neural ode model, we need to pass the appropriate learnable parameters, `parameters` which is
returned by the `construct_models` function. It is simply the `node.p` vector. We then train our model
for `20` epochs.

```julia
model, parameters = construct_model(1, 2, 64, 0)

for _ in 1:20
    Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)
end
```

Here is what the contour plot should look for Neural ODE. Notice that the regression is not perfect due to
the thin artifact which connects the circles.

![node](https://user-images.githubusercontent.com/30564094/85356368-6d96a880-b52c-11ea-8f21-6f35df0d5c8c.png)

## Training the Augmented Neural ODE

Our training configuration will be same as that of Neural ODE. Only in this case we have augmented the
input with a single zero. This makes the problem 3 dimensional and as such it is possible to find
a function which can be expressed by the neural ode. For more details and proofs please refer to \[1].

```julia
model, parameters = construct_model(1, 2, 64, 1)

for _ in 1:20
    Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)
end
```

For the augmented Neural ODE we notice that the artifact is gone.

![anode](https://user-images.githubusercontent.com/30564094/85356373-6f606c00-b52c-11ea-8431-fb74ff01dde5.png)

# Expected Output

```julia
Generating Dataset
Training Neural ODE
Iteration 10 || Loss = 1.120950346042287
Iteration 20 || Loss = 0.7532827576978123
Iteration 30 || Loss = 0.6553072657563143
Iteration 40 || Loss = 0.5776224441286926
Iteration 50 || Loss = 0.520220032887143
Iteration 60 || Loss = 0.4720985558909219
Iteration 70 || Loss = 0.40864928280982776
Iteration 80 || Loss = 0.31852414526416173
Iteration 90 || Loss = 0.2481226283169072
Iteration 100 || Loss = 0.18494126827346924
Iteration 110 || Loss = 0.1470610323283855
Iteration 120 || Loss = 0.11047625817186725
Iteration 130 || Loss = 0.08345815290524315
Iteration 140 || Loss = 0.07403708346807107
Iteration 150 || Loss = 0.07533284314926621
Iteration 160 || Loss = 0.0807687251841976
Iteration 170 || Loss = 0.07462567382624537
Iteration 180 || Loss = 0.07620629179132667
Iteration 190 || Loss = 0.053866679112776615
Iteration 200 || Loss = 0.04878556792331758
Iteration 210 || Loss = 0.04951108534243689
Iteration 220 || Loss = 0.05219504247382631
Iteration 230 || Loss = 0.04829135367843049
Iteration 240 || Loss = 0.049767843225925146
Iteration 250 || Loss = 0.053435584980049196
Iteration 260 || Loss = 0.06330034970941116
Iteration 270 || Loss = 0.0674349462469368
Iteration 280 || Loss = 0.054646060311423855
Iteration 290 || Loss = 0.058587195675343276

Training Augmented Neural ODE
Iteration 10 || Loss = 0.8052180670411724
Iteration 20 || Loss = 0.6493323407007557
Iteration 30 || Loss = 0.43219055806073575
Iteration 40 || Loss = 0.17996087680619746
Iteration 50 || Loss = 0.10677325033978695
Iteration 60 || Loss = 0.06763209051354825
Iteration 70 || Loss = 0.05557440341534089
Iteration 80 || Loss = 0.05175036274021895
Iteration 90 || Loss = 0.04598351412837163
Iteration 100 || Loss = 0.043676742887634665
Iteration 110 || Loss = 0.03984149571984987
Iteration 120 || Loss = 0.0377291626995965
Iteration 130 || Loss = 0.03618670060916931
Iteration 140 || Loss = 0.034320209919086186
Iteration 150 || Loss = 0.0336497946533457
Iteration 160 || Loss = 0.031927051142359046
Iteration 170 || Loss = 0.029903758740715262
Iteration 180 || Loss = 0.027844496982240865
Iteration 190 || Loss = 0.026784325989841085
Iteration 200 || Loss = 0.024220633742520606
Iteration 210 || Loss = 0.022410493158517744
Iteration 220 || Loss = 0.021468741537086423
Iteration 230 || Loss = 0.022414650405633993
Iteration 240 || Loss = 0.018358236240712273
Iteration 250 || Loss = 0.01700812985852846
Iteration 260 || Loss = 0.01573442645516374
Iteration 270 || Loss = 0.014445097490699855
Iteration 280 || Loss = 0.014170123422892231
Iteration 290 || Loss = 0.01250998329088748
```

# References

\[1] Dupont, Emilien, Arnaud Doucet, and Yee Whye Teh. "Augmented neural odes." Advances in Neural Information Processing Systems. 2019.
