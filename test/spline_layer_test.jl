using DiffEqFlux, Flux
using LinearAlgebra, Distributions
using Optim
using Test
using DataInterpolations

function run_test(f, layer, atol)

    data_train_vals = rand(500)
    data_train_fn = f.(data_train_vals)

    function loss_function(θ)
        data_pred = [layer(x, θ) for x in data_train_vals]
        loss = sum(abs.(data_pred.-data_train_fn))/length(data_train_fn)
        return loss
    end

    function cb(p,l)
        @show l
        return false
    end

    res = DiffEqFlux.sciml_train(loss_function, layer.saved_points, ADAM(0.1), cb=cb, maxiters = 100)
    res = DiffEqFlux.sciml_train(loss_function, res.minimizer, ADAM(0.01), cb=cb, maxiters = 100)
    opt = res.minimizer

    data_validate_vals = rand(100)
    data_validate_fn = f.(data_validate_vals)

    data_validate_pred = [layer(x,opt) for x in data_validate_vals]

    return sum(abs.(data_validate_pred.-data_validate_fn))/length(data_validate_fn) < atol
end

ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:dl}) = A.dl,y -> Tridiagonal(dl,zeros(length(d)),zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:d}) = A.d,y -> Tridiagonal(zeros(length(dl)),d,zeros(length(du)),)
ZygoteRules.@adjoint ZygoteRules.literal_getproperty(A::Tridiagonal, ::Val{:du}) = A.dl,y -> Tridiagonal(zeros(length(dl)),zeros(length(d),du),)
ZygoteRules.@adjoint Tridiagonal(dl, d, du) = Tridiagonal(dl, d, du), p̄ -> (diag(p̄[2:end,1:end-1]),diag(p̄),diag(p̄[1:end-1,2:end]))

##test 01: affine function, Linear Interpolation
a, b = rand(2)
f = x -> a*x + b
layer = SplineLayer((0.0,1.0),0.01,LinearInterpolation)
@test run_test(f, layer, 0.05)

##test 02: non-linear function, Quadratic Interpolation
a, b, c = rand(3)
f = x -> a*x^2+ b*x + x
layer = SplineLayer((0.0,1.0),0.01,QuadraticInterpolation)
@test run_test(f, layer, 0.05)

##test 03: non-linear function, Quadratic Spline
a, b, c = rand(3)
f = x -> a*sin(b*x+c)
layer = SplineLayer((0.0,1.0),0.1,QuadraticSpline)
@test run_test(f, layer, 0.05)

##test 03: non-linear function, Cubic Spline
a, b, c = rand(3)
f = x -> exp(x)*x^2
layer = SplineLayer((0.0,1.0),0.1,CubicSpline)
@test run_test(f, layer, 0.05)
