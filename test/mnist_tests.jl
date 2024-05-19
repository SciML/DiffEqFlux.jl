@testsetup module MNISTTestSetup

using Reexport

@reexport using DiffEqFlux, CUDA, Zygote, MLDataUtils, NNlib, OrdinaryDiffEq, Test, Lux,
                Statistics, ComponentArrays, Random, Optimization, OptimizationOptimisers,
                LuxCUDA
@reexport using MLDatasets: MNIST
@reexport using MLDataUtils: LabelEnc, convertlabel, stratifiedobs

CUDA.allowscalar(false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

const cdev = cpu_device()
const gdev = gpu_device()

logitcrossentropy(ŷ, y) = mean(-sum(y .* logsoftmax(ŷ; dims = 1); dims = 1))

function loadmnist(batchsize = bs)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    function onehot(labels_raw)
        convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
    end
    # Load MNIST
    mnist = MNIST(; split = :train)
    imgs, labels_raw = mnist.features, mnist.targets
    # Process images into (H,W,C,BS) batches
    x_train = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3))) |>
              gdev
    x_train = batchview(x_train[:, :, :, 1:(10 * batchsize)], batchsize)
    # Onehot and batch the labels
    y_train = onehot(labels_raw) |> gdev
    y_train = batchview(y_train[:, 1:(10 * batchsize)], batchsize)
    return x_train, y_train
end

const bs = 128
x_train, y_train = loadmnist(bs)

function DiffEqArray_to_Array(x)
    return reduce((x, y) -> cat(x, y; dims = ndims(first(x.u))), x.u)
end

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 100)
    total_correct = 0
    total = 0
    st = Lux.testmode(st)
    for (x, y) in collect(data)[1:min(n_batches, length(data))]
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function loss_function(m, ps, x, y, st)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

export x_train, y_train, DiffEqArray_to_Array, gdev, cdev, classify, accuracy, loss_function

end

@testitem "MNIST Neural ODE MLP" tags=[:cuda] skip=:(using CUDA; !CUDA.functional()) setup=[MNISTTestSetup] begin
    down = Chain(FlattenLayer(), Dense(784, 20, tanh))
    nn = Chain(Dense(20, 10, tanh), Dense(10, 10, tanh), Dense(10, 20, tanh))
    fc = Dense(20, 10)

    nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
        reltol = 1e-3, abstol = 1e-3, save_start = false)

    m = Chain(; down, nn_ode, convert = WrappedFunction(DiffEqArray_to_Array), fc)
    ps, st = Lux.setup(Xoshiro(0), m)
    ps = ComponentArray(ps) |> gdev
    st = st |> gdev

    #We can also build the model topology without a NN-ODE
    m_no_ode = Lux.Chain(; down, nn, fc)
    ps_no_ode, st_no_ode = Lux.setup(Xoshiro(0), m_no_ode)
    ps_no_ode = ComponentArray(ps_no_ode) |> gdev
    st_no_ode = st_no_ode |> gdev

    #To understand the intermediate NN-ODE layer, we can examine it's dimensionality
    x_d = first(down(x_train[1], ps.down, st.down))

    # We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
    x_m = first(m(x_train[1], ps, st))
    #Or without the NN-ODE layer.
    x_m = first(m_no_ode(x_train[1], ps_no_ode, st_no_ode))

    # burn in accuracy
    accuracy(m, zip(x_train, y_train), ps, st)

    # burn in loss
    loss_function(m, ps, x_train[1], y_train[1], st)

    opt = OptimizationOptimisers.Adam(0.05)
    iter = 0

    opt_func = OptimizationFunction(
        (ps, _, x, y) -> loss_function(m, ps, x, y, st), Optimization.AutoZygote())
    opt_prob = OptimizationProblem(opt_func, ps)

    function callback(ps, l, pred)
        global iter += 1
        #Monitor that the weights do infact update
        #Every 10 training iterations show accuracy
        if (iter % 10 == 0)
            @info "[MNIST GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps.u, st))"
        end
        return false
    end

    # Train the NN-ODE and monitor the loss and weights.
    res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); callback)
    @test accuracy(m, zip(x_train, y_train), res.u, st) > 0.7
end

@testitem "MNIST Neural ODE Conv" tags=[:cuda] skip=:(using CUDA; !CUDA.functional()) setup=[MNISTTestSetup] timeout=3600 begin
    down = Chain(Conv((3, 3), 1 => 64, relu; stride = 1), GroupNorm(64, 8),
        Conv((4, 4), 64 => 64, relu; stride = 2, pad = 1),
        GroupNorm(64, 8), Conv((4, 4), 64 => 64, relu; stride = 2, pad = 1))

    dudt = Chain(Conv((3, 3), 64 => 64, relu; stride = 1, pad = 1),
        Conv((3, 3), 64 => 64, relu; stride = 1, pad = 1))

    fc = Chain(GroupNorm(64, 8, relu), MeanPool((6, 6)), FlattenLayer(), Dense(64, 10))

    nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
        reltol = 1e-3, abstol = 1e-3, save_start = false, dt = 0.1f0)

    # Build our over-all model topology
    m = Chain(down,           # (28, 28, 1, BS) -> (6, 6, 64, BS)
        nn_ode,               # (6, 6, 64, BS) -> (6, 6, 64, BS, 1)
        DiffEqArray_to_Array, # (6, 6, 64, BS, 1) -> (6, 6, 64, BS)
        fc)                   # (6, 6, 64, BS) -> (10, BS)
    ps, st = Lux.setup(Xoshiro(0), m)
    ps = ComponentArray(ps) |> gdev
    st = st |> gdev

    # To understand the intermediate NN-ODE layer, we can examine it's dimensionality
    img = x_train[1][:, :, :, 1:1] |> gdev
    lab = y_train[2][:, 1:1] |> gdev

    x_m, _ = m(img, ps, st)

    # burn in accuracy
    accuracy(m, zip(x_train, y_train), ps, st)

    # burn in loss
    loss_function(m, ps, x_train[1], y_train[1], st)

    opt = OptimizationOptimisers.Adam(0.05)
    iter = 0

    opt_func = OptimizationFunction(
        (ps, _, x, y) -> loss_function(m, ps, x, y, st), Optimization.AutoZygote())

    opt_prob = OptimizationProblem(opt_func, ps)

    function callback(ps, l, pred)
        global iter += 1
        #Monitor that the weights do infact update
        #Every 10 training iterations show accuracy
        if (iter % 10 == 0)
            @info "[MNIST Conv] Accuracy: $(accuracy(m, zip(x_train, y_train), ps.u, st))"
        end
        return false
    end

    # Train the NN-ODE and monitor the loss and weights.
    res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); maxiters = 10, callback)
    @test accuracy(m, zip(x_train, y_train), res.u, st) > 0.7
end
