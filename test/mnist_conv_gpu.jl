using DiffEqFlux, CUDA, Zygote, MLDatasets, OrdinaryDiffEq, Printf, Test
using Flux.Losses: logitcrossentropy
using Flux.Data: DataLoader
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

function loadmnist(batchsize = bs, train_split = 0.9)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    mnist = MNIST(split = :train)
    imgs, labels_raw = mnist.features, mnist.targets
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

down = Flux.Chain(Conv((3, 3), 1=>64, relu, stride = 1), GroupNorm(64, 64),
             Conv((4, 4), 64=>64, relu, stride = 2, pad=1), GroupNorm(64, 64),
             Conv((4, 4), 64=>64, stride = 2, pad = 1)) |>gpu

dudt = Flux.Chain(Conv((3, 3), 64=>64, tanh, stride=1, pad=1),
             Conv((3, 3), 64=>64, tanh, stride=1, pad=1)) |>gpu

fc = Flux.Chain(GroupNorm(64, 64), x -> relu.(x), MeanPool((6, 6)),
           x -> reshape(x, (64, :)), Flux.Dense(64,10)) |> gpu

nn_ode = NeuralODE(dudt, (0.f0, 1.f0), Tsit5(),
                   save_everystep = false,
                   reltol = 1e-3, abstol = 1e-3,
                   save_start = false) |> gpu

function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return xarr[:,:,:,:,1]
end

# Build our over-all model topology
model = Flux.Chain(down,                 # (28, 28, 1, BS) -> (6, 6, 64, BS)
              nn_ode,               # (6, 6, 64, BS) -> (6, 6, 64, BS, 1)
              DiffEqArray_to_Array, # (6, 6, 64, BS, 1) -> (6, 6, 64, BS)
              fc)                   # (6, 6, 64, BS) -> (10, BS)

# To understand the intermediate NN-ODE layer, we can examine it's dimensionality
img, lab = train_dataloader.data[1][:, :, :, 1:1], train_dataloader.data[2][:, 1:1]

x_d = down(img)

# We can see that we can compute the forward pass through the NN topology
# featuring an NNODE layer.
x_m = model(img)

classify(x) = argmax.(eachcol(x))

function accuracy(model, data; n_batches = 10)
    total_correct = 0
    total = 0
    for (i, (x, y)) in enumerate(data)
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

Flux.train!(loss, Flux.params(down, nn_ode.p, fc), train_dataloader, opt, cb = cb)
@test accuracy(model, test_dataloader; n_batches = length(test_dataloader)) > 0.8
