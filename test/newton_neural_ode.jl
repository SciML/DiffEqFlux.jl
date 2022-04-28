using DiffEqFlux, GalacticOptim, GalacticOptimJL, OrdinaryDiffEq, Random, Test

Random.seed!(100)

n = 1 # number of ODEs
tspan = (0.0, 1.0)

d = 5 # number of data pairs
x = rand(n, 5)
y = rand(n, 5)

cb = function (p,l)
  @show l
  false
end

NN = Chain(Dense(n, 5n, tanh),
           Dense(5n, n))

@info "ROCK4"
nODE = NeuralODE(NN, tspan, ROCK4(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ) = Flux.Losses.mse(y, nODE(x, θ)[end])
l1 = loss_function(nODE.p)

res = DiffEqFlux.sciml_train(loss_function, nODE.p, NewtonTrustRegion(), GalacticOptim.AutoZygote(), maxiters = 100, cb=cb)
@test loss_function(res.minimizer) < l1
res = DiffEqFlux.sciml_train(loss_function, nODE.p, Optim.KrylovTrustRegion(), GalacticOptim.AutoZygote(), maxiters = 100, cb=cb)
@test loss_function(res.minimizer) < l1

NN = FastChain(FastDense(n, 5n, tanh),
               FastDense(5n, n))

@info "ROCK2"
nODE = NeuralODE(NN, tspan, ROCK2(), reltol=1e-4, saveat=[tspan[end]])

loss_function(θ) = Flux.Losses.mse(y, nODE(x, θ)[end])
l1 = loss_function(nODE.p)
optfunc = GalacticOptim.OptimizationFunction((x, p) -> loss_function(x), GalacticOptim.AutoZygote())
optprob = GalacticOptim.OptimizationProblem(optfunc, nODE.p,)

res = GalacticOptim.solve(optprob, NewtonTrustRegion(), maxiters = 100, cb=cb)
@test loss_function(res.minimizer) < l1
res = GalacticOptim.solve(optprob, Optim.KrylovTrustRegion(), maxiters = 100, cb=cb)
@test loss_function(res.minimizer) < l1
