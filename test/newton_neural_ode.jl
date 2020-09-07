using DiffEqFlux, Flux, Optim, OrdinaryDiffEq, Test

n = 2  # number of ODEs
tspan = (0.0, 1.0)

d = 5  # number of data pairs
x = rand(n, 5)
y = rand(n, 5)

cb = function (p,l)
  @show l
  false
end

NN = Chain(Dense(n, 10n, tanh),
           Dense(10n, n))

@info "ROCK4"
nODE = NeuralODE(NN, tspan, ROCK4(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ) = Flux.mse(y, nODE(x, θ))
l1 = loss_function(nODE.p)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, Optim.KrylovTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1

NN = FastChain(FastDense(n, 10n, tanh),
               FastDense(10n, n))

@info "ROCK2"
nODE = NeuralODE(NN, tspan, ROCK2(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ) = Flux.mse(y, nODE(x, θ))
l1 = loss_function(nODE.p)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, Optim.KrylovTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1

@info "ROCK4"
nODE = NeuralODE(NN, tspan, ROCK4(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ) = Flux.mse(y, nODE(x, θ))
l1 = loss_function(nODE.p)
res = DiffEqFlux.sciml_train(loss_function, nODE.p, LBFGS(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, Optim.KrylovTrustRegion(), maxiters = 100, cb=cb)
@test 10loss_function(res.minimizer) < l1
