using DiffEqFlux, OrdinaryDiffEq, Flux, MLDataUtils, NNlib
using Flux: logitcrossentropy
using MLDatasets: MNIST
using CUDA
CUDA.allowscalar(false)

function loadmnist(batchsize = bs)
	# Use MLDataUtils LabelEnc for natural onehot conversion
  	onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:9)))
	# Load MNIST
	imgs, labels_raw = MNIST.traindata();
	# Process images into (H,W,C,BS) batches
	x_train = Float32.(reshape(imgs,size(imgs,1),size(imgs,2),1,size(imgs,3))) |> gpu
	x_train = batchview(x_train,batchsize)
	# Onehot and batch the labels
	y_train = onehot(labels_raw) |> gpu
	y_train = batchview(y_train,batchsize)
	return x_train, y_train
end

# Main
const bs = 128
x_train, y_train = loadmnist(bs)

down = Chain(x->reshape(x,(28*28,:)),
             Dense(784,20,tanh)
            ) |> gpu
nfe = 0
nn  = Chain(
           Dense(20,10,tanh),
           Dense(10,10,tanh),
           Dense(10,20,tanh)
          ) |> gpu
fc  = Chain( Dense(20,10) ) |> gpu

nn_ode = NeuralODE( nn, (0.f0, 1.f0), Tsit5(),
                        save_everystep = false,
                        reltol = 1e-3, abstol = 1e-3,
                        save_start = false ) |> gpu

"""
    DiffEqArray_to_Array( x )

Cheap conversion of a `DiffEqArray` instance to a Matrix.
"""
function DiffEqArray_to_Array( x )
    xarr = gpu( x )
    return reshape( xarr, size(xarr)[1:2] )
end

#Build our over-all model topology
m        = Chain(down,
                 nn_ode,
                 DiffEqArray_to_Array,
                 fc ) |> gpu
#We can also build the model topology without a NN-ODE
m_no_ode = Chain( down, nn, fc) |> gpu

#To understand the intermediate NN-ODE layer, we can examine it's dimensionality
x_d = down( x_train[1] )

# We can see that we can compute the forward pass through the NN topology featuring an NNODE layer.
x_m = m(x_train[1])
#Or without the NN-ODE layer.
x_m = m_no_ode(x_train[1])

classify(x) = argmax.(eachcol(x))

function accuracy(model,data; n_batches=100)
    total_correct = 0
    total = 0
    for (x,y) in collect(data)[1:n_batches]
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(model(x)))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct/total
end
#burn in accuracy
accuracy(m, zip(x_train,y_train))

loss(x,y) = logitcrossentropy(m(x),y)
#burn in loss
loss(x_train[1],y_train[1])

opt = ADAM(0.05)
iter = 0

cb() = begin
    global iter += 1
    #Monitor that the weights do infact update
    #Every 10 training iterations show accuracy
    (iter%10 == 0) && @show accuracy(m, zip(x_train,y_train))
    global nfe=0
end

# Train the NN-ODE and monitor the loss and weights.
Flux.train!( loss, params( down, nn_ode.p, fc), zip( x_train, y_train ), opt, cb = cb )
