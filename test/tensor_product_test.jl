using DiffEqFlux, Flux
using LinearAlgebra, Distributions
using Optim
using Test

function run_test(f, layer, atol)

    data_train_vals = [rand(length(layer.model)) for k in 1:500]
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

    optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_function(x), layer.p, GalacticOptim.AutoForwardDiff())
    optprob = GalacticOptim.OptimizationProblem(optfunc, layer.p)
    res = GalacticOptim.solve(optprob, ADAM(0.1), cb=cb, maxiters = 100)
    optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    res = GalacticOptim.solve(optprob, ADAM(0.01), cb=cb, maxiters = 100)
    optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    res = GalacticOptim.solve(optprob, BFGS(), cb=cb, maxiters = 200)
    opt = res.minimizer

    data_validate_vals = [rand(length(layer.model)) for k in 1:100]
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    return sum(norm.(data_validate_pred.-data_validate_fn))/length(data_validate_fn) < atol
end

##test 01: affine function, Chebyshev and Polynomial basis
A = rand(2,2)
b = rand(2)
f = x -> A*x + b
layer = TensorLayer([ChebyshevBasis(10), PolynomialBasis(10)], 2)
@test run_test(f, layer, 0.05)

##test 02: non-linear function, Chebyshev and Legendre basis
A = rand(2,2)
b = rand(2)
f = x -> A*x*norm(x)+ b*sin(norm(x))
layer = TensorLayer([ChebyshevBasis(7), FourierBasis(7)], 2)
@test run_test(f, layer, 0.10)
