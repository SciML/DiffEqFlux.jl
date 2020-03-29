using Flux, Optim, OrdinaryDiffEq, RecursiveArrayTools, DifferentialEquations
include("./DiffEqFlux.jl")

function main()
  n = 5  # number of ODEs
  tspan = (0.0, 1.0)

  x_data = rand(n, 5)
  y_data = rand(n, 5)


  training_data = Flux.Data.DataLoader(x_data, y_data, batchsize=1)

  NN = Chain(Dense(n, 10n, tanh),
             Dense(10n, n))


  function cb(p, l, x, pred)
    println("This is the size of the output of nODE ", size(pred))
    false
  end

  nODE = DiffEqFlux.NeuralODEBatchConv(NN, tspan, ROCK4(),
                              reltol=1e-4, saveat=[tspan[end]])

  function loss_function(θ, x, y)
    pred = nODE(x, θ)
    Flux.mse(y, pred), x, pred
  end
  res = DiffEqFlux.sciml_train(loss_function, nODE.p, ADAM(0.01), training_data,
                               cb=cb, save_best=false, progress=true)
end
@timev main()
