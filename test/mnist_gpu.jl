using DiffEqFlux, CUDA, Zygote, MLDataUtils, NNlib, OrdinaryDiffEq, Test, Lux, Statistics,
      ComponentArrays, Random, Optimization, OptimizationOptimisers, LuxCUDA
using MLDatasets: MNIST
using MLDataUtils: LabelEnc, convertlabel, stratifiedobs

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
    x_train = batchview(x_train, batchsize)
    # Onehot and batch the labels
    y_train = onehot(labels_raw) |> gdev
    y_train = batchview(y_train, batchsize)
    return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Lux.Chain(Lux.FlattenLayer(), Lux.Dense(784, 20, tanh))
nn = Lux.Chain(Lux.Dense(20, 10, tanh), Lux.Dense(10, 10, tanh),
    Lux.Dense(10, 20, tanh))
fc = Lux.Dense(20, 10)

nn_ode = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(); save_everystep = false, reltol = 1e-3,
    abstol = 1e-3, save_start = false)

"""
    DiffEqArray_to_Array(x)

Cheap conversion of a `DiffEqArray` instance to a Matrix.
"""
function DiffEqArray_to_Array(x)
    xarr = gdev(x)
    return reshape(xarr, size(xarr)[1:2])
end

#Build our over-all model topology
m = Lux.Chain(; down, nn_ode, convert = Lux.WrappedFunction(DiffEqArray_to_Array), fc)
ps, st = Lux.setup(Random.default_rng(), m)
ps = ComponentArray(ps) |> gdev
st = st |> gdev

#We can also build the model topology without a NN-ODE
m_no_ode = Lux.Chain(; down, nn, fc)
ps_no_ode, st_no_ode = Lux.setup(Random.default_rng(), m_no_ode)
ps_no_ode = ComponentArray(ps_no_ode) |> gdev
st_no_ode = st_no_ode |> gdev

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = first(down(x_train[1], ps.down, st.down))

# We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = first(m(x_train[1], ps, st))
#Or without the NN-ODE layer.
x_m = first(m_no_ode(x_train[1], ps_no_ode, st_no_ode))

classify(x) = argmax.(eachcol(x))

function accuracy(model, data, ps, st; n_batches = 100)
    total_correct = 0
    total = 0
    st = Lux.testmode(st)
    for (x, y) in collect(data)[1:n_batches]
        target_class = classify(cdev(y))
        predicted_class = classify(cdev(first(model(x, ps, st))))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end
#burn in accuracy
accuracy(m, zip(x_train, y_train), ps, st)

function loss_function(ps, x, y)
    pred, st_ = m(x, ps, st)
    return logitcrossentropy(pred, y), pred
end

#burn in loss
loss_function(ps, x_train[1], y_train[1])

opt = OptimizationOptimisers.Adam(0.05)
iter = 0

opt_func = OptimizationFunction((ps, _, x, y) -> loss_function(ps, x, y),
    Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps)

function callback(ps, l, pred)
    global iter += 1
    #Monitor that the weights do infact update
    #Every 10 training iterations show accuracy
    if (iter % 10 == 0)
        @info "[MNIST GPU] Accuracy: $(accuracy(m, zip(x_train, y_train), ps, st))"
    end
    return false
end

# Train the NN-ODE and monitor the loss and weights.
res = Optimization.solve(opt_prob, opt, zip(x_train, y_train); callback)
@test accuracy(m, zip(x_train, y_train), res.u, st) > 0.8
