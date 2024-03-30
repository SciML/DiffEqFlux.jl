using DiffEqFlux, Distributions, Zygote, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, LinearAlgebra, Random, ComponentArrays, Test

function run_test(f, layer, atol, N)
    ps, st = Lux.setup(Xoshiro(0), layer)
    ps = ComponentArray(ps)
    model = Lux.Experimental.StatefulLuxLayer(layer, ps, st)

    data_train_vals = [rand(N) for k in 1:500]
    data_train_fn = f.(data_train_vals)

    function loss_function(p)
        data_pred = [model(x, p) for x in data_train_vals]
        loss = sum(norm.(data_pred .- data_train_fn)) / length(data_train_fn)
        return loss
    end

    function cb(p, l)
        @info "[TensorProductLayer] Loss: $l"
        return false
    end

    optfunc = Optimization.OptimizationFunction((x, p) -> loss_function(x),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optfunc, ps)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.1); callback = cb, maxiters = 100)
    optprob = Optimization.OptimizationProblem(optfunc, res.minimizer)
    res = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01); callback = cb, maxiters = 100)
    optprob = Optimization.OptimizationProblem(optfunc, res.minimizer)
    res = Optimization.solve(optprob, BFGS(); callback = cb, maxiters = 200)
    opt = res.minimizer

    data_validate_vals = [rand(N) for k in 1:100]
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [model(x, opt) for x in data_validate_vals]

    return sum(norm.(data_validate_pred .- data_validate_fn)) / length(data_validate_fn) <
           atol
end

##test 01: affine function, Chebyshev and Polynomial basis
A = rand(2, 2)
b = rand(2)
layer = TensorLayer([ChebyshevBasis(10), PolynomialBasis(10)], 2)
@test run_test(x -> A * x + b, layer, 0.05, 2)

##test 02: non-linear function, Chebyshev and Legendre basis
A = rand(2, 2)
b = rand(2)
layer = TensorLayer([ChebyshevBasis(7), FourierBasis(7)], 2)
@test run_test(x -> A * x * norm(x) + b * sin(norm(x)), layer, 0.10, 2)
