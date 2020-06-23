using DiffEqFlux, Flux
using LinearAlgebra, Distributions
using Optim
using Test

function run_test(f, layer::TensorLayer, atol)

    data_train_vals = [rand(length(layer.model)) for k in 1:100]
    data_train_fn = f.(data_train_vals)

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
    res = DiffEqFlux.sciml_train(loss_function, res.minimizer, ADAM(0.005), cb=cb, maxiters = 100)
    res = DiffEqFlux.sciml_train(loss_function, res.minimizer, ADAM(0.001), cb=cb, maxiters = 100)
    res = DiffEqFlux.sciml_train(loss_function, res.minimizer, BFGS(), cb=cb, maxiters = 100)
    opt = res.minimizer

    data_validate_vals = [rand(length(layer.model)) for k in 1:100]
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    return sum(norm.(data_validate_pred.-data_validate_fn))/length(data_train_fn) < atol
end

##test 01: affine function, Chebyshev and Polynomial basis
A = rand(2,2)
b = rand(2)
f = x -> A*x + b
layer = TensorLayer([ChebyshevBasis(10), PolynomialBasis(10)], 2)
@test run_test(f, layer, 0.05)

##test 02: non-linear function, Legendre and Fourier basis
A = rand(2,2)
b = rand(2)
f = x -> A*x*norm(x)+ b/norm(x)
layer = TensorLayer([LegendreBasis(20), ChebyshevBasis(20)], 2)
@test run_test(f, layer2, 0.15)
