using DiffEqFlux, Flux, Random
using LinearAlgebra, Distributions
using Optim, ComponentArrays

data_train_vals = []

for i in 1:100
    x = [rand() rand()]
    push!(data_train_vals, x)
end

f = x -> [x[1]^2, x[1]*x[2]]

data_train_fn = f.(data_train_vals)

basis1 = ChebyshevBasis(6)
basis2 =PolynomialBasis(4)
M = [basis1, basis2]

layer = TensorLayer(M, 2)

function loss_function(component)
    data_pred = [layer(x,component) for x in data_train_vals]
    loss = sum(norm.(data_pred.-data_train_fn))/length(data_train_fn)
    return loss
end

function cb(p,l)
    @show l
    return false
end

res = DiffEqFlux.sciml_train(loss_function, layer.p, ADAM(0.1), cb=cb, maxiters = 100)
opt = res.minimizer
res = DiffEqFlux.sciml_train(loss_function, opt, LBFGS(), cb=cb, maxiters = 100)
opt = res.minimizer

data_validate_vals = []
for i in 1:100
    x = [rand() rand()]
    push!(data_validate_vals, x)
end

data_validate_fn = f.(data_validate_vals)
data_validate_pred = [layer(x,opt) for x in data_validate_vals]

@show sum(norm.(data_validate_pred.-data_validate_fn))/length(data_train_fn)

data_train_vals = []

for i in 1:100
    x = [rand() rand()]
    push!(data_train_vals, x)
end

f = x -> [x[2]^2, x[1]/x[2]]

data_train_fn = f.(data_train_vals)

basis1 = ChebyshevBasis(6)
basis2 =PolynomialBasis(4)
M = [basis1, basis2]

layer = TensorLayer(M, 2)

function loss_function(component)
    data_pred = [layer(x,component) for x in data_train_vals]
    loss = sum(norm.(data_pred.-data_train_fn))/length(data_train_fn)
    return loss
end

function cb(p,l)
    @show l
    return false
end

res = DiffEqFlux.sciml_train(loss_function, layer.p, ADAM(0.1), cb=cb, maxiters = 100)
opt = res.minimizer
res = DiffEqFlux.sciml_train(loss_function, opt, LBFGS(), cb=cb, maxiters = 100)
opt = res.minimizer

data_validate_vals = []
for i in 1:100
    x = [rand() rand()]
    push!(data_validate_vals, x)
end

data_validate_fn = f.(data_validate_vals)
data_validate_pred = [layer(x,opt) for x in data_validate_vals]

@show sum(norm.(data_validate_pred.-data_validate_fn))/length(data_train_fn)
