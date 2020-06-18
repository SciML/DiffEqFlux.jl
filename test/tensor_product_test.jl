using DiffEqFlux, Flux, Random
using LinearAlgebra, Distributions
using Optim, ComponentArrays

data_train_vals = []

μ = zeros(8)
Σ = I + zeros(8,8)
for i in 1:100
    x = rand(MvNormal(μ, Σ))
    push!(data_train_vals, x)
end

A = rand(2, 8)
b = rand(2)
f = x -> A*x./(norm(x)) + b.*norm(x)

data_train_fn = f.(data_train_vals)

basis1 = ChebyshevPolyBasis(4)
basis2 = LegendrePolyBasis(4)
basis3 = FourierBasis(4)
basis4 = PolynomialBasis(4)
basis5 = ChebyshevPolyBasis(4)
basis6 = LegendrePolyBasis(4)
basis7 = FourierBasis(4)
basis8 = PolynomialBasis(4)
A = [basis1, basis2, basis3, basis4, basis5, basis6, basis7, basis8]

layer = TensorLayer(A, 2)

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

data_validate_vals = []

for i in 1:100
    x = rand(MvNormal(μ, Σ))
    push!(data_validate_vals, x)
end

A = rand(2, 8)
b = rand(2)
f = x -> A*x./(norm(x)) + b.*norm(x)

data_validate_fn = f.(data_validate_vals)
data_validate_pred = [layer(x,opt) for x in data_validate_vals]

@show sum(norm.(data_validate_pred.-data_validate_fn))/length(data_train_fn)
