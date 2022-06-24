# Neural ODEs on GPUs

Note that the differential equation solvers will run on the GPU if the initial
condition is a GPU array. Thus, for example, we can define a neural ODE by hand
that runs on the GPU (if no GPU is available, the calculation defaults back to the CPU):

```julia
using DifferentialEquations, Flux, DiffEqFlux, DiffEqSensitivity

using Random
rng = Random.default_rng()

model_gpu = Chain(Dense(2, 50, tanh), Dense(50, 2)) |> gpu
p, re = Flux.destructure(model_gpu)
dudt!(u, p, t) = re(p)(u)

# Simulation interval and intermediary points
tspan = (0f0, 10f0)
tsteps = 0f0:1f-1:10f0

u0 = Float32[2.0; 0.0] |> gpu
prob_gpu = ODEProblem(dudt!, u0, tspan, p)

# Runs on a GPU
sol_gpu = solve(prob_gpu, Tsit5(), saveat = tsteps)
```

Or we could directly use the neural ODE layer function, like:

```julia
prob_neuralode_gpu = NeuralODE(gpu(model_gpu), tspan, Tsit5(), saveat = tsteps)
```

If one is using `Lux.Chain`, then the computation takes place on the GPU with
`f(x,p,st)` if `x`, `p` and `st` are on the GPU. This commonly looks like:

```julia
import Lux

dudt2 = Lux.Chain(x -> x.^3,
            Lux.Dense(2,50,tanh),
            Lux.Dense(50,2))

u0 = Float32[2.; 0.] |> gpu
p, st = Lux.setup(rng, dudt2) .|> gpu

dudt2_(u, p, t) = dudt2(u,p,st)[1]

# Simulation interval and intermediary points
tspan = (0f0, 10f0)
tsteps = 0f0:1f-1:10f0

prob_gpu = ODEProblem(dudt2_, u0, tspan, p)

# Runs on a GPU
sol_gpu = solve(prob_gpu, Tsit5(), saveat = tsteps)
```

or via the NeuralODE struct:

```julia
prob_neuralode_gpu = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
prob_neuralode_gpu(u0,p,st)
```

## Neural ODE Example

Here is the full neural ODE example. Note that we use the `gpu` function so that the
same code works on CPUs and GPUs, dependent on `using CUDA`.

```julia
using Flux, DiffEqFlux, Optimization, OptimizationFlux, Zygote, 
      OrdinaryDiffEq, Plots, CUDA, DiffEqSensitivity, Random, ComponentArrays
CUDA.allowscalar(false) # Makes sure no slow operations are occuring

#rng for Lux.setup
rng = Random.default_rng()
# Generate Data
u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
# Make the data into a GPU-based array if the user has a GPU
ode_data = gpu(solve(prob_trueode, Tsit5(), saveat = tsteps))


dudt2 = Chain(x -> x.^3, Dense(2, 50, tanh), Dense(50, 2)) |> gpu
u0 = Float32[2.0; 0.0] |> gpu
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  gpu(prob_neuralode(u0,p))
end
function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end
# Callback function to observe training
list_plots = []
iter = 0
callback = function (p, l, pred; doplot = false)
  global list_plots, iter
  if iter == 0
    list_plots = []
  end
  iter += 1
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, Array(ode_data[1,:]), label = "data")
  scatter!(plt, tsteps, Array(pred[1,:]), label = "prediction")
  push!(list_plots, plt)
  if doplot
    display(plot(plt))
  end
  return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p)->loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, prob_neuralode.p)
result_neuralode = Optimization.solve(optprob,ADAM(0.05),callback = callback,maxiters = 300)
```
