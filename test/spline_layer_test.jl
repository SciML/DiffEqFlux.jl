using DiffEqFlux, DataInterpolations, Distributions, GalacticOptim, LinearAlgebra, Test

function run_test(f, layer, atol)

    data_train_vals = rand(500)
    data_train_fn = f.(data_train_vals)

    function loss_function(θ)
        data_pred = [layer(x, θ) for x in data_train_vals]
        loss = sum(abs.(data_pred.-data_train_fn))/length(data_train_fn)
        return loss
    end

    function callback(p,l)
        @show l
        return false
    end

    optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_function(x), GalacticOptim.AutoZygote())
    optprob = GalacticOptim.OptimizationProblem(optfunc, layer.saved_points)
    res = GalacticOptim.solve(optprob, ADAM(0.1), callback=callback, maxiters = 100)

    optprob = GalacticOptim.OptimizationProblem(optfunc, res.minimizer)
    res = GalacticOptim.solve(optprob, ADAM(0.1), callback=callback, maxiters = 100)
    opt = res.minimizer

    data_validate_vals = rand(100)
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    output = sum(abs.(data_validate_pred.-data_validate_fn))/length(data_validate_fn)
    @show output
    return output < atol
end

##test 01: affine function, Linear Interpolation
a, b = rand(2)
f = x -> a*x + b
layer = SplineLayer((0.0,1.0),0.01,LinearInterpolation)
@test run_test(f, layer, 0.1)

##test 02: non-linear function, Quadratic Interpolation
a, b, c = rand(3)
f = x -> a*x^2+ b*x + x
layer = SplineLayer((0.0,1.0),0.01,QuadraticInterpolation)
@test run_test(f, layer, 0.1)

##test 03: non-linear function, Quadratic Spline
a, b, c = rand(3)
f = x -> a*sin(b*x+c)
layer = SplineLayer((0.0,1.0),0.1,QuadraticSpline)
@test run_test(f, layer, 0.1)

##test 04: non-linear function, Cubic Spline
f = x -> exp(x)*x^2
layer = SplineLayer((0.0,1.0),0.1,CubicSpline)
@test run_test(f, layer, 0.1)
