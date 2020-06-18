using DiffEqFlux, Flux, Random
using LinearAlgebra
using Optim, ComponentArrays

data_train_vals = []

for i in 1:100
    x = [rand() rand()]
    push!(data_train_vals, x)
end

f = x -> [x[1]+x[2], x[1]-x[2]]
data_train_fn = f.(data_train_vals)

basis1 = PolynomialBasis(2)
basis2 = LegendrePolyBasis(2)
A = [basis1, basis2]
layer = TensorProductLayer(A, 2)

function loss_function(component)
    data_pred = [layer(x,component) for x in data_train_vals]
    loss = sum(norm.(data_pred.-data_train_fn))/length(data_train_fn)
    return loss
end

function cb(p,l)
    @show l
    return false
end

res = DiffEqFlux.sciml_train(loss_function, layer.component, ADAM(0.1), cb=cb, maxiters = 100)
res = DiffEqFlux.sciml_train(loss_function, res.minimizer, LBFGS(), cb=cb)

component_opt = res.minimizer
x = [7.0 4.0]
layer(x, component_opt)
