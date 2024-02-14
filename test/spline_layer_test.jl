using DiffEqFlux, ComponentArrays, Zygote, DataInterpolations, Distributions, Optimization,
      OptimizationOptimisers, LinearAlgebra, Random, Test

function run_test(f, layer, atol)
    ps, st = Lux.setup(Xoshiro(0), layer)
    ps = ComponentArray(ps)
    model = Lux.Experimental.StatefulLuxLayer(layer, ps, st)

    data_train_vals = rand(500)
    data_train_fn = f.(data_train_vals)

    function loss_function(θ)
        data_pred = [model(x, θ) for x in data_train_vals]
        loss = sum(abs.(data_pred .- data_train_fn)) / length(data_train_fn)
        return loss
    end

    function callback(p, l)
        @info "[SplineLayer] Loss: $l"
        return false
    end

    optfunc = Optimization.OptimizationFunction((x, p) -> loss_function(x),
        Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optfunc, ps)
    res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 100)

    optprob = Optimization.OptimizationProblem(optfunc, res.minimizer)
    res = Optimization.solve(optprob, Adam(0.1); callback = callback, maxiters = 100)
    opt = res.minimizer

    data_validate_vals = rand(100)
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [model(x, opt) for x in data_validate_vals]

    output = sum(abs.(data_validate_pred .- data_validate_fn)) / length(data_validate_fn)
    return output < atol
end

##test 01: affine function, Linear Interpolation
a, b = rand(2)
layer = SplineLayer((0.0, 1.0), 0.01, LinearInterpolation)
@test run_test(x -> a * x + b, layer, 0.1)

##test 02: non-linear function, Quadratic Interpolation
a, b, c = rand(3)
layer = SplineLayer((0.0, 1.0), 0.01, QuadraticInterpolation)
@test run_test(x -> a * x^2 + b * x + x, layer, 0.1)

##test 03: non-linear function, Quadratic Spline
a, b, c = rand(3)
layer = SplineLayer((0.0, 1.0), 0.1, QuadraticSpline)
@test run_test(x -> a * sin(b * x + c), layer, 0.1)

##test 04: non-linear function, Cubic Spline
layer = SplineLayer((0.0, 1.0), 0.1, CubicSpline)
@test run_test(x -> exp(x) * x^2, layer, 0.1)
