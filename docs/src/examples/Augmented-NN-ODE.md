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

@info "Generating Dataset"

dataloader = concentric_sphere(2, (0.0, 2.0), (3.0, 4.0), 2000, 2000; batch_size = 256)

model, parameters = construct_model(1, 2, 64, 0)
opt = ADAM(0.005)
iter = 0

cb = function()
    global iter += 1
    if iter % 10 == 1
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end

@info "Training Neural ODE"

Flux.@epochs 20 Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)

plt_node = plot_contour(model)

model, parameters = construct_model(1, 2, 64, 1)
opt = ADAM(0.005)
iter = 0

cb = function()
    global iter += 1
    if iter % 10 == 0
        println("Iteration $iter || Loss = $(loss_node(dataloader.data[1], dataloader.data[2]))")
    end
end

@info "Training Augmented Neural ODE"

Flux.@epochs 20 Flux.train!(loss_node, Flux.Params([parameters]), dataloader, opt, cb = cb)

plt_anode = plot_contour(model)
```

# Expected Output

```julia
┌ Info: Generating Dataset
└ @ Main In[11]:60
┌ Info: Training Neural ODE
└ @ Main In[11]:75
┌ Info: Epoch 1
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 1 || Loss = 5.027746376044498
Iteration 11 || Loss = 1.120950346042287
┌ Info: Epoch 2
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 21 || Loss = 0.7532827576978123
┌ Info: Epoch 3
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 31 || Loss = 0.6553072657563143
Iteration 41 || Loss = 0.5776224441286926
┌ Info: Epoch 4
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 51 || Loss = 0.520220032887143
┌ Info: Epoch 5
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 61 || Loss = 0.4720985558909219
Iteration 71 || Loss = 0.40864928280982776
┌ Info: Epoch 6
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 81 || Loss = 0.31852414526416173
┌ Info: Epoch 7
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 91 || Loss = 0.2481226283169072
Iteration 101 || Loss = 0.18494126827346924
┌ Info: Epoch 8
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 111 || Loss = 0.1470610323283855
┌ Info: Epoch 9
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 121 || Loss = 0.11047625817186725
Iteration 131 || Loss = 0.08345815290524315
┌ Info: Epoch 10
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 141 || Loss = 0.07403708346807107
┌ Info: Epoch 11
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 151 || Loss = 0.07533284314926621
Iteration 161 || Loss = 0.0807687251841976
┌ Info: Epoch 12
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 171 || Loss = 0.07462567382624537
┌ Info: Epoch 13
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 181 || Loss = 0.07620629179132667
Iteration 191 || Loss = 0.053866679112776615
┌ Info: Epoch 14
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 201 || Loss = 0.04878556792331758
┌ Info: Epoch 15
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 211 || Loss = 0.04951108534243689
Iteration 221 || Loss = 0.05219504247382631
┌ Info: Epoch 16
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 231 || Loss = 0.04829135367843049
┌ Info: Epoch 17
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 241 || Loss = 0.049767843225925146
Iteration 251 || Loss = 0.053435584980049196
┌ Info: Epoch 18
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 261 || Loss = 0.06330034970941116
┌ Info: Epoch 19
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 271 || Loss = 0.0674349462469368
Iteration 281 || Loss = 0.054646060311423855
┌ Info: Epoch 20
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 291 || Loss = 0.058587195675343276
┌ Info: Training Augmented Neural ODE
└ @ Main In[11]:92
┌ Info: Epoch 1
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 10 || Loss = 0.8052180670411724
┌ Info: Epoch 2
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 20 || Loss = 0.6493323407007557
Iteration 30 || Loss = 0.43219055806073575
┌ Info: Epoch 3
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 40 || Loss = 0.17996087680619746
┌ Info: Epoch 4
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 50 || Loss = 0.10677325033978695
Iteration 60 || Loss = 0.06763209051354825
┌ Info: Epoch 5
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 70 || Loss = 0.05557440341534089
┌ Info: Epoch 6
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 80 || Loss = 0.05175036274021895
Iteration 90 || Loss = 0.04598351412837163
┌ Info: Epoch 7
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 100 || Loss = 0.043676742887634665
┌ Info: Epoch 8
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 110 || Loss = 0.03984149571984987
Iteration 120 || Loss = 0.0377291626995965
┌ Info: Epoch 9
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 130 || Loss = 0.03618670060916931
┌ Info: Epoch 10
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 140 || Loss = 0.034320209919086186
Iteration 150 || Loss = 0.0336497946533457
┌ Info: Epoch 11
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 160 || Loss = 0.031927051142359046
┌ Info: Epoch 12
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 170 || Loss = 0.029903758740715262
Iteration 180 || Loss = 0.027844496982240865
┌ Info: Epoch 13
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 190 || Loss = 0.026784325989841085
┌ Info: Epoch 14
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 200 || Loss = 0.024220633742520606
Iteration 210 || Loss = 0.022410493158517744
┌ Info: Epoch 15
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 220 || Loss = 0.021468741537086423
┌ Info: Epoch 16
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 230 || Loss = 0.022414650405633993
Iteration 240 || Loss = 0.018358236240712273
┌ Info: Epoch 17
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 250 || Loss = 0.01700812985852846
┌ Info: Epoch 18
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 260 || Loss = 0.01573442645516374
Iteration 270 || Loss = 0.014445097490699855
┌ Info: Epoch 19
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 280 || Loss = 0.014170123422892231
┌ Info: Epoch 20
└ @ Main /home/avik-pal/.julia/packages/Flux/Fj3bt/src/optimise/train.jl:121
Iteration 290 || Loss = 0.01250998329088748
```