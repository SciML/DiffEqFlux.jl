# Data-Parallel Multithreaded, Distributed, and Multi-GPU Batching

DiffEqFlux.jl allows for data-parallel batching optimally on one
computer, across an entire compute cluster, and batching along GPUs.
This can be done by parallelizing within an ODE solve or between the
ODE solves. The automatic differentiation tooling is compatible with
the parallelism. The following examples demonstrate training over a few
different modes of parallelism. These examples are not exhaustive.

## Within-ODE Multithreaded and GPU Batching

We end by noting that there is an alternative way of batching which
can be more efficient in some cases like neural ODEs. With a neural
networks, columns are treated independently (by the properties of
matrix multiplication). Thus for example, with `FastChain` we can
define an ODE:

```julia
using DiffEqFlux, OrdinaryDiffEq

dudt = FastChain(FastDense(2,50,tanh),FastDense(50,2))
p = initial_params(dudt)
f(u,p,t) = dudt(u,p)
```

and we can solve this ODE where the initial condition is a vector:

```julia
u0 = Float32[2.; 0.]
prob = ODEProblem(f,u0,(0f0,1f0),p)
solve(prob,Tsit5())
```

or we can solve this ODE where the initial condition is a matrix, where
each column is an independent system:

```julia
u0 = Float32.([0 1 2
               0 0 0])
prob = ODEProblem(f,u0,(0f0,1f0),p)
solve(prob,Tsit5())
```

On the CPU this will multithread across the system (due to BLAS) and
on GPUs this will parallelize the operations across the GPU. To GPU
this, you'd simply move the parameters and the initial condition to the
GPU:

```julia
xs = Float32.([0 1 2
               0 0 0])
prob = ODEProblem(f,gpu(u0),(0f0,1f0),gpu(p))
solve(prob,Tsit5())
```

This method of parallelism is optimal if all of the operations are
linear algebra operations such as a neural ODE. Thus this method of
parallelism is demonstrated in the [MNIST tutorial](@ref mnist).

However, this method of parallelism has many limitations. First of all,
the ODE function is required to be written in a way that is independent
across the columns. Not all ODEs are written like this, so one needs to
be careful. But additionally, this method is ineffective if the ODE
function has many serial operations, like `u[1]*u[2] - u[3]`. In such
a case, this indexing behavior will dominate the runtime and cause the
parallelism to sometimes even be detrimental.

# Out of ODE Parallelism

Instead of parallelizing within an ODE solve, one can parallelize the
solves to the ODE itself. While this will be less effective on very
large ODEs, like big neural ODE image classifiers, this method be effective
even if the ODE is small or the `f` function is not well-parallelized.
This kind of parallelism is done via the [DifferentialEquations.jl ensemble interface](https://diffeq.sciml.ai/stable/features/ensemble/). The following examples
showcase multithreaded, cluster, and (multi)GPU parallelism through this
interface.

## Multithreaded Batching At a Glance

The following is a full copy-paste example for the multithreading.
Distributed and GPU minibatching are described below.

```julia
using OrdinaryDiffEq, DiffEqSensitivity, DiffEqFlux
pa = [1.0]
u0 = [3.0]
θ = [u0;pa]

function model1(θ,ensemble)
  prob = ODEProblem((u, p, t) -> 1.01u .* p, [θ[1]], (0.0, 1.0), [θ[2]])

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = 0.1, trajectories = 100)
end

# loss function
loss_serial(θ)   = sum(abs2,1.0.-Array(model1(θ,EnsembleSerial())))
loss_threaded(θ) = sum(abs2,1.0.-Array(model1(θ,EnsembleThreads())))

cb = function (θ,l) # callback function to observe training
  @show l
  false
end

opt = ADAM(0.1)
l1 = loss_serial(θ)
res_serial = DiffEqFlux.sciml_train(loss_serial, θ, opt; cb = cb, maxiters=100)
res_threads = DiffEqFlux.sciml_train(loss_threaded, θ, opt; cb = cb, maxiters=100)
```

## Multithreaded Batching In-Depth

In order to make use of the ensemble interface, we need to build an
`EnsembleProblem`. The `prob_func` is the function for determining
the different `DEProblem`s to solve. This is the place where we can
randomly sample initial conditions or pull initial conditions from
an array of batches in order to perform our study. To do this, we
first define a prototype `DEProblem`. Here we use the following
`ODEProblem` as our base:

```julia
prob = ODEProblem((u, p, t) -> 1.01u .* p, [θ[1]], (0.0, 1.0), [θ[2]])
```

In the `prob_func` we define how to build a new problem based on the
base problem. In this case, we want to change `u0` by a constant, i.e.
`0.5 .+ i/100 .* prob.u0` for different trajectories labelled by `i`.
Thus we use the [remake function from the problem interface](https://diffeq.sciml.ai/stable/basics/problem/#Modification-of-problem-types) to do so:

```julia
function prob_func(prob, i, repeat)
  remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
end
```

We now build the `EnsembleProblem` with this basis:

```julia
ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
```

Now to solve an ensemble problem, we need to choose an ensembling
algorithm and choose the number of trajectories to solve. Here let's
solve this in serial with 100 trajectories. Note that `i` will thus run
from `1:100`.

```julia
sim = solve(ensemble_prob, Tsit5(), EnsembleSerial(), saveat = 0.1, trajectories = 100)
```

and thus running in multithreading would be:

```julia
sim = solve(ensemble_prob, Tsit5(), EnsembleThreads(), saveat = 0.1, trajectories = 100)
```

This whole mechanism is differentiable, so we then put it in a training
loop and it soars. Note that you need to make sure that [Julia's multithreading](https://docs.julialang.org/en/v1/manual/multi-threading/)
is enabled, which you can do via:

```julia
Threads.nthreads()
```

## Distributed Batching Across a Cluster

Changing to distributed computing is very simple as well. The setup is
all the same, except you utilize `EnsembleDistributed` as the ensembler:

```julia
sim = solve(ensemble_prob, Tsit5(), EnsembleDistributed(), saveat = 0.1, trajectories = 100)
```

Note that for this to work you need to ensure that your processes are
already started. For more information on setting up processes and utilizing
a compute cluster, see [the official distributed documentation](https://docs.julialang.org/en/v1/manual/distributed-computing/). The key feature to recognize is that, due to
the message passing required for cluster compute, one needs to ensure
that all of the required functions are defined on the worker processes.
The following is a full example of a distributed batching setup:

```julia
using Distributed
addprocs(4)

@everywhere begin
  using OrdinaryDiffEq, DiffEqSensitivity, Flux, DiffEqFlux
  function f(u,p,t)
    1.01u .* p
  end
end

pa = [1.0]
u0 = [3.0]
θ = [u0;pa]

function model1(θ,ensemble)
  prob = ODEProblem(f, [θ[1]], (0.0, 1.0), [θ[2]])

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = 0.1, trajectories = 100)
end

cb = function (θ,l) # callback function to observe training
  @show l
  false
end

opt = ADAM(0.1)
loss_distributed(θ) = sum(abs2,1.0.-Array(model1(θ,EnsembleDistributed())))
l1 = loss_distributed(θ)
res_distributed = DiffEqFlux.sciml_train(loss_distributed, θ, opt; cb = cb, maxiters=100)
```

And note that only `addprocs(4)` needs to be changed in order to make
this demo run across a cluster. For more information on adding processes
to a cluster, check out [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl).

## Minibatching Across GPUs with DiffEqGPU

DiffEqGPU.jl allows for generating code parallelizes an ensemble on
generated CUDA kernels. This method is efficient for sufficiently
small (<100 ODE) problems where the significant computational cost
is due to the large number of batch trajectories that need to be
solved. This kernel-building process adds a few restrictions to the
function, such as requiring it has no boundschecking or allocations.
The following is an example of minibatch ensemble parallelism across
a GPU:

```julia
using OrdinaryDiffEq, DiffEqSensitivity, Flux, DiffEqFlux
function f(du,u,p,t)
  @inbounds begin
    du[1] = 1.01 * u[1] * p[1] * p[2]
  end
end

pa = [1.0]
u0 = [3.0]
θ = [u0;pa]

function model1(θ,ensemble)
  prob = ODEProblem(f, [θ[1]], (0.0, 1.0), [θ[2]])

  function prob_func(prob, i, repeat)
    remake(prob, u0 = 0.5 .+ i/100 .* prob.u0)
  end

  ensemble_prob = EnsembleProblem(prob, prob_func = prob_func)
  sim = solve(ensemble_prob, Tsit5(), ensemble, saveat = 0.1, trajectories = 100)
end

cb = function (θ,l) # callback function to observe training
  @show l
  false
end

opt = ADAM(0.1)
loss_gpu(θ) = sum(abs2,1.0.-Array(model1(θ,EnsembleGPUArray())))
l1 = loss_gpu(θ)
res_gpu = DiffEqFlux.sciml_train(loss_gpu, θ, opt; cb = cb, maxiters=100)
```

## Multi-GPU Batching

DiffEqGPU supports batching across multiple GPUs. See [its README](https://github.com/SciML/DiffEqGPU.jl#setting-up-multi-gpu)
for details on setting it up.
